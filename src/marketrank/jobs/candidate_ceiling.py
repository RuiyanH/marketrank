"""
Step R.5 -- build the candidate sources, measure the ceiling and each source's
marginal contribution at a fixed slot budget.

    python -m marketrank.jobs.candidate_ceiling --ann artifacts/twotower/runs/<run>/ann_candidates.parquet

THE COHORT IS FIXED. The same 20,000 hash-sampled `val_tune` customers and the
same 70,715 true (customer, article) pairs every other number in this build is
measured on. The job asserts the denominator rather than trusting it -- a
ceiling measured on a different cohort is not comparable to the recall numbers
it is supposed to bound, and that mistake is silent.

Sources, each tagged with its origin:

  repurchase     the customer's own prior distinct articles       (week 4)
  category_pop   popularity inside the customer's dominant type   (week 4)
  global_pop     plain global recent popularity                   (R.5)
  covisit        time-decayed item-item co-visitation             (R.5)
  ann            the two-tower's top-N, if a parquet is supplied  (week 3/R.4)

The output table is the source-contribution table R.5's checkpoint asks for and
the input to R.6's keep/demote rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyspark.sql import functions as F

from marketrank import candidates as C, covisit, splits
from marketrank.retrieval import baselines as B

EXPECT_PAIRS = 70_715
EXPECT_CUSTOMERS = 20_000

# Args that describe HOW A SOURCE WAS DERIVED. On the --sources-from path these
# describe nothing: the parquets were built by an earlier run, possibly at other
# settings, and argparse still fills these with defaults. Leaving them in the
# artifact makes it assert bounds it never used -- and covisit's defaults (60/50)
# are the ones measured as disk-fatal here, so the artifact would name the
# configuration that crashes as the one that produced the numbers.
DERIVATION_ARGS = (
    "n_repurchase", "n_category", "n_global_pop", "n_covisit",
    "covisit_lookback", "covisit_max_basket", "ann", "no_category",
)

# Which of them built which source -- the sidecar's contents.
SOURCE_ARGS = {
    "repurchase": ("n_repurchase",),
    "category_pop": ("n_category",),
    "global_pop": ("n_global_pop",),
    "covisit": ("n_covisit", "covisit_lookback", "covisit_max_basket"),
    "ann": ("ann",),
}


def source_meta_path(parquet: Path) -> Path:
    return parquet.with_suffix(".meta.json")


def write_source_meta(parquet: Path, name: str, args, as_of: str, rows: int) -> None:
    """
    Provenance beside the parquet, written when the source is materialised.

    The sidecar is the durable half of the fix: a parquet that travels without
    its settings is unauditable, and `artifacts/` is gitignored, so the JSON's
    self-description is all a regenerated artifact has. This matters most for
    runs produced on the cluster, which someone other than their author may read.
    """
    source_meta_path(parquet).write_text(
        json.dumps(
            {
                "source": name,
                "as_of": as_of,
                "rows": rows,
                "args": {
                    k: (str(v) if isinstance(v, Path) else v)
                    for k, v in vars(args).items()
                    if k in SOURCE_ARGS.get(name, ())
                },
            },
            indent=2, default=str,
        )
    )


def read_source_meta(parquet: Path) -> dict | None:
    mp = source_meta_path(parquet)
    if not mp.exists():
        return None
    return json.loads(mp.read_text())


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ann", type=Path, default=None, help="ANN candidate parquet")
    p.add_argument("--slice", default="val_tune")
    p.add_argument("--n-repurchase", type=int, default=C.N_REPURCHASE)
    p.add_argument("--n-category", type=int, default=C.N_CATEGORY)
    p.add_argument("--n-global-pop", type=int, default=C.N_GLOBAL_POP)
    p.add_argument("--n-covisit", type=int, default=C.N_COVISIT)
    # Co-visitation's cost bounds, exposed because the defaults do not fit a
    # laptop. MEASURED at as_of=2020-08-12: lookback 60 / basket 50 is 443,559
    # customers and a 35.7M-pair self-join, and each pair carries a 64-char
    # customer_id that gets shuffled twice -- it exhausted a 13 GiB disk. At
    # 30 / 20 the same join is 9.0M pairs. Both are modelling statements as much
    # as cost knobs: see covisit.py's module docstring.
    p.add_argument("--covisit-lookback", type=int, default=covisit.LOOKBACK_DAYS)
    p.add_argument("--covisit-max-basket", type=int, default=covisit.MAX_BASKET)
    p.add_argument("--no-category", action="store_true")
    # Re-measure a different SUBSET of an existing run without recomputing any
    # source. The sources are written to parquet by every run (see the
    # materialisation note below), so a question like "what is the ceiling
    # without category_pop" is a union over five small tables, not a rebuild.
    p.add_argument(
        "--sources-from",
        type=Path,
        default=None,
        help="load sources from a previous run's sources/ dir instead of deriving them",
    )
    p.add_argument(
        "--drop",
        action="append",
        default=[],
        metavar="SOURCE",
        help="drop a source by name; repeatable",
    )
    p.add_argument("--out", type=Path, default=Path("artifacts/candidates"))
    p.add_argument("--expect-pairs", type=int, default=EXPECT_PAIRS)
    return p.parse_args(argv)


def main(argv=None) -> dict:
    from marketrank.spark import get_spark

    args = parse_args(argv)
    spark = get_spark("candidate_ceiling", driver_memory="10g")

    lo, _hi = splits.bounds(args.slice)
    as_of = lo

    # The cohort comes from the exported eval set, so it is provably the same
    # 20,000 customers the tower is evaluated on rather than a re-derivation.
    cohort = spark.read.parquet(
        str(Path("artifacts/twotower/eval_customers"))
    ).select("customer_id").distinct()
    truth = B.truth_pairs(spark, args.slice).join(
        F.broadcast(cohort), "customer_id", "inner"
    )

    n_cust, n_pairs = cohort.count(), truth.count()
    print(f"COHORT customers {n_cust}  true_pairs {n_pairs}")
    if args.expect_pairs and n_pairs != args.expect_pairs:
        raise SystemExit(
            f"denominator is {n_pairs}, expected {args.expect_pairs} -- this "
            "ceiling would not be comparable to any other number in the build"
        )

    sources: dict = {}

    loaded_meta: dict = {}
    if args.sources_from is not None:
        # Canonical order so `names` -- and therefore every leave-one-out --
        # is stable across runs and comparable run to run.
        order = [
            C.SOURCE_REPURCHASE, C.SOURCE_CATEGORY, C.SOURCE_GLOBAL_POP,
            C.SOURCE_COVISIT, C.SOURCE_ANN,
        ]
        for name in order:
            path = args.sources_from / f"{name}.parquet"
            if path.exists() and name not in args.drop:
                sources[name] = spark.read.parquet(str(path))
                meta = read_source_meta(path)
                loaded_meta[name] = meta
                shown = meta["args"] if meta else "NO SIDECAR -- provenance unknown"
                print(f"LOADED {name:<14} <- {path}  {shown}")
        if not sources:
            raise SystemExit(f"no source parquets found under {args.sources_from}")
        return _measure(
            spark, args, sources, truth,
            loaded_from=args.sources_from, source_meta=loaded_meta,
        )

    sources[C.SOURCE_REPURCHASE] = C.repurchase_source(
        spark, as_of, n=args.n_repurchase
    ).join(F.broadcast(cohort), "customer_id", "inner")

    if not args.no_category:
        sources[C.SOURCE_CATEGORY] = C.category_popularity_source(
            spark, as_of, n=args.n_category
        ).join(F.broadcast(cohort), "customer_id", "inner")

    sources[C.SOURCE_GLOBAL_POP] = C.cross_to_customers(
        C.global_popularity_source(spark, as_of, n=args.n_global_pop), cohort
    )

    sources[C.SOURCE_COVISIT] = covisit.covisit_source(
        spark,
        as_of,
        customers=cohort,
        n=args.n_covisit,
        lookback_days=args.covisit_lookback,
        max_basket=args.covisit_max_basket,
    )

    if args.ann is not None and args.ann.exists():
        sources[C.SOURCE_ANN] = (
            spark.read.parquet(str(args.ann))
            .join(F.broadcast(cohort), "customer_id", "inner")
            .select("customer_id", "article_id", "source", "source_rank")
        )

    for name in args.drop:
        sources.pop(name, None)

    names = tuple(sources)
    print("SOURCES", names)

    # ------------------------------------------------------------------------
    # MATERIALISE EVERY SOURCE BEFORE MEASURING. Not an optimisation -- without
    # it this job cannot finish on a laptop, and the reason is worth stating.
    #
    # `marginal_contribution` runs ~22 Spark actions over 5 sources: a solo
    # ceiling and a slot count each, a leave-one-out union and slot count each,
    # plus the full union twice. Every action that touches `covisit` re-derives
    # its self-join from raw transactions, because a Spark DataFrame is a plan,
    # not a table. Co-visitation appears in ~12 of those actions.
    #
    # MEASURED: that recomputation retained 5.2 GB of shuffle files for the
    # session's lifetime and exhausted the disk twice -- ~450 MB per derivation
    # x ~12. Tightening covisit's bounds from 60/50 to 30/20 cut the join 4x
    # (35.7M pairs -> 9.0M) and did NOT fix it, which is what identified
    # recomputation rather than join size as the cause.
    #
    # Writing each source to parquet costs seconds and a few MB -- every source
    # is at most `n` rows per customer over 20,000 customers -- and turns 12
    # derivations into 1. It also leaves the sources on disk to inspect, which
    # is how R.6's decision gets audited rather than trusted.
    # ------------------------------------------------------------------------
    src_dir = args.out / "sources"
    src_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = src_dir / f"{name}.parquet"
        sources[name].write.mode("overwrite").parquet(str(path))
        sources[name] = spark.read.parquet(str(path))
        n_rows = sources[name].count()
        write_source_meta(path, name, args, as_of, n_rows)
        print(f"MATERIALISED {name:<14} rows {n_rows:>9} -> {path}")

    return _measure(spark, args, sources, truth)


def _measure(
    spark, args, sources: dict, truth,
    loaded_from: Path | None = None, source_meta: dict | None = None,
) -> dict:
    """Union, ceiling, marginals, and the artifact -- shared by both entry paths."""
    names = tuple(sources)
    union = C.union_candidates(*sources.values(), source_names=names)
    union.cache()
    stats = C.candidate_set_stats(union, source_names=names)
    ceiling = C.recall_ceiling(union, truth, source_names=names)
    marg = C.marginal_contribution(
        sources, truth, budget_note=f"{stats['mean_candidates_per_customer']:.1f} cand/customer"
    )

    print("STATS", json.dumps(stats, indent=2, default=str))
    print("CEILING", json.dumps(ceiling, indent=2))
    print("MARGINAL", json.dumps(marg, indent=2))

    # PROVENANCE. Without this the artifact cannot be reproduced from itself:
    # the covisit bounds actually used, the ANN parquet consumed, `as_of` and
    # the per-source depths all lived only in the shell history. Worse, the CLI
    # DEFAULTS (60/50) are settings this build measured as disk-fatal on this
    # machine, so the docstring's example command reproduces the crash rather
    # than the artifact. `run` is what makes the numbers auditable.
    lo, _hi = splits.bounds(args.slice)
    derived = loaded_from is None
    all_args = {
        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
    }
    run = {
        "as_of": lo,
        "slice": args.slice,
        "sources": list(names),
        "loaded_from": str(loaded_from) if loaded_from else None,
        "derived": derived,
        # On the loaded path the derivation args are argparse defaults that
        # describe nothing. They are moved aside rather than echoed, because a
        # reader who trusts `args` would otherwise attribute these numbers to
        # covisit 60/50 -- the settings that exhaust this machine's disk.
        "derivation_args_used": derived,
        "args": (
            all_args if derived
            else {k: v for k, v in all_args.items() if k not in DERIVATION_ARGS}
        ),
        "source_provenance": (
            {n: (source_meta or {}).get(n) for n in names} if not derived else None
        ),
    }
    if not derived:
        run["derivation_args_ignored"] = {
            k: v for k, v in all_args.items() if k in DERIVATION_ARGS
        }

    args.out.mkdir(parents=True, exist_ok=True)
    out_name = args.out / "ceiling.json"
    out_name.write_text(
        json.dumps(
            {"run": run, "stats": stats, "ceiling": ceiling, "marginal": marg},
            indent=2, default=str,
        )
    )
    print(f"WROTE {out_name}")

    # The table R.5's checkpoint asks for, and R.6's criterion reads.
    print("\nSOURCE CONTRIBUTION AT FIXED BUDGET "
          f"({stats['mean_candidates_per_customer']:.1f} candidates/customer)")
    print(f"{'source':<14}{'solo':>9}{'reach':>8}{'marginal':>10}{'slots':>9}{'marg/slot':>12}")
    for name, d in sorted(
        marg["sources"].items(), key=lambda kv: -kv[1]["marginal_per_slot"]
    ):
        print(
            f"{name:<14}{d['solo']*100:>8.3f}%{d['reach_frac']*100:>7.1f}%"
            f"{d['marginal']*100:>9.3f}%"
            f"{d['marginal_slots']:>9.1f}{d['marginal_per_slot']*100:>11.4f}%"
        )
    print(f"{'UNION':<14}{ceiling['recall_ceiling']*100:>8.3f}%")

    spark.stop()
    return {"run": run, "stats": stats, "ceiling": ceiling, "marginal": marg}


if __name__ == "__main__":
    main()
