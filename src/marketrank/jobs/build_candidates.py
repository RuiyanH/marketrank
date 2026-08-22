"""
Candidate generation at ranker scale, per-day `as_of` -- C1's orchestrator.

    # gate the whole job on zero GPU work, before generating anything
    python -m marketrank.jobs.build_candidates --checksum-day \\
        --ann-snapshot artifacts/twotower/runs/r2_recency/ann_candidates.parquet

    # the real run
    python -m marketrank.jobs.build_candidates --lo-day 90 --hi-day 691

**THE CHECKSUM COMES FIRST, AND IT IS DECOMPOSED.** Restricting this machinery
to the `val_tune` cohort on day 692 must reproduce the shipped table exactly --
not just the 11.930% union, but all five per-source solos. A union that matches
tells you nothing broke; five solos tell you WHICH source moved when something
does. The reference is read from the shipped artifact
(`artifacts/candidates_misha_90_50/ceiling.json`, tracked in git as of B2)
rather than transcribed into this file, so the number the checksum asserts and
the number the README quotes cannot drift apart.

It costs no GPU time, because the shipped `r2_recency` ANN parquet IS the day-692
snapshot -- see `--ann-snapshot` below. So this gates the 692-day run before the
expensive stage exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, functions as F

from marketrank import candidates as C, candidates_daily as CD, config
from marketrank.retrieval import baselines as B

SHIPPED_CEILING = Path("artifacts/candidates_misha_90_50/ceiling.json")

# What the shipped ANN snapshot must look like. Asserted on read rather than
# trusted: this file is the checksum's only unverified input.
ANN_SNAPSHOT_ROWS = 1_000_000
ANN_SNAPSHOT_CUSTOMERS = 20_000
ANN_SNAPSHOT_TOP_N = 50

SOURCE_ORDER = (
    C.SOURCE_REPURCHASE,
    C.SOURCE_CATEGORY,
    C.SOURCE_GLOBAL_POP,
    C.SOURCE_COVISIT,
    C.SOURCE_ANN,
)

# Tolerance on the checksum. The per-day machinery is deterministic and should
# reproduce the shipped numbers to the pair, so this is a float-comparison
# guard, not a margin for "close enough". A real regression moves recall by
# orders of magnitude more than this.
CHECKSUM_TOL = 1e-9


# ---------------------------------------------------------------------------
# The ANN contract. ONE definition, exercised by the checksum today and binding
# on the GPU stage when it exists -- so the expensive path is validated by the
# same assertion the cheap path already runs, rather than by a second one
# written later and never tested.
# ---------------------------------------------------------------------------
ANN_COLUMNS = ("customer_id", "day_index", "article_id", "source_rank")


def assert_ann_contract(df: DataFrame, top_n: int = ANN_SNAPSHOT_TOP_N) -> DataFrame:
    """Every ANN producer -- snapshot or per-day GPU -- returns this shape."""
    missing = [c for c in ANN_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"ANN output is missing columns {missing}; got {df.columns}")
    bad = df.filter(
        (F.col("source_rank") < 1) | (F.col("source_rank") > top_n)
    ).limit(1).count()
    if bad:
        raise SystemExit(f"ANN output has source_rank outside 1..{top_n}")
    return df.select(*ANN_COLUMNS)


def load_ann_snapshot(
    spark: SparkSession, path: Path, day: int, strict: bool = True
) -> tuple[DataFrame, dict]:
    """
    Read a SINGLE-DAY ANN parquet and stamp it to `day`.

    A CHECKSUM FIXTURE, NOT AN INPUT. The shipped `r2_recency` export is the
    day-692 object by construction -- customer features as of d-1 = 2020-08-11,
    `as_of` = `val_tune`'s first day -- so stamping it to 692 is legitimate.
    Stamping it across a multi-day run would score every day with 2020-08-12's
    retrieval, which is the per-day-ANN leak in its purest form. The caller
    enforces the single-day restriction; this function records what it read.
    """
    df = spark.read.parquet(str(path))
    n_rows = df.count()
    n_cust = df.select("customer_id").distinct().count()
    max_rank = df.agg(F.max("source_rank")).collect()[0][0]

    if strict:
        problems = []
        if n_rows != ANN_SNAPSHOT_ROWS:
            problems.append(f"rows {n_rows} != {ANN_SNAPSHOT_ROWS}")
        if n_cust != ANN_SNAPSHOT_CUSTOMERS:
            problems.append(f"customers {n_cust} != {ANN_SNAPSHOT_CUSTOMERS}")
        if max_rank is None or max_rank > ANN_SNAPSHOT_TOP_N:
            problems.append(f"max source_rank {max_rank} > {ANN_SNAPSHOT_TOP_N}")
        if problems:
            raise SystemExit(
                f"ANN snapshot at {path} is not the shipped day-{day} object: "
                + "; ".join(problems)
            )

    meta = {
        "path": str(path),
        "sha256": _content_hash(path),
        "rows": n_rows,
        "customers": n_cust,
        "max_source_rank": int(max_rank) if max_rank is not None else None,
        "stamped_day": int(day),
    }
    stamped = df.withColumn("day_index", F.lit(int(day)))
    return assert_ann_contract(stamped), meta


def _content_hash(path: Path) -> str:
    """sha256 over the file, or over every part in deterministic order."""
    h = hashlib.sha256()
    files = [path] if path.is_file() else sorted(
        p for p in path.rglob("*") if p.is_file() and not p.name.startswith(".")
    )
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def build_sources(
    spark: SparkSession,
    events: DataFrame,
    pairs: DataFrame,
    phase: int,
    cadence: int = CD.COVISIT_CADENCE_DAYS,
    ann: DataFrame | None = None,
    n_repurchase: int = 30,
    n_category: int = 40,
    n_global_pop: int = 40,
    n_covisit: int = 60,
    recent_k: int = 10,
    covisit_lookback: int = 90,
    covisit_max_basket: int = 50,
) -> dict[str, DataFrame]:
    """All five sources in one shape: (customer_id, day_index, article_id, source_rank)."""
    out: dict[str, DataFrame] = {}

    out[C.SOURCE_REPURCHASE] = CD.daily_repurchase(spark, events, n=n_repurchase)

    dom = CD.daily_dominant_category(spark, events)
    cat = CD.daily_category_pop(spark, n=n_category)
    out[C.SOURCE_CATEGORY] = dom.join(
        cat, ["day_index", "product_type_no"], "inner"
    ).select("customer_id", "day_index", "article_id", "source_rank")

    gp = CD.daily_global_pop(spark, n=n_global_pop)
    out[C.SOURCE_GLOBAL_POP] = events.join(gp, "day_index", "inner").select(
        "customer_id", "day_index", "article_id", "source_rank"
    )

    out[C.SOURCE_COVISIT] = CD.daily_covisit(
        spark, events, pairs, n=n_covisit, recent_k=recent_k,
        lookback_days=covisit_lookback, max_basket=covisit_max_basket,
        cadence_days=cadence, phase=phase,
    )

    if ann is not None:
        out[C.SOURCE_ANN] = ann
    return out


def _as_single_day(df: DataFrame, day: int) -> DataFrame:
    """Drop `day_index` so `candidates`' single-day helpers apply unchanged."""
    return df.filter(F.col("day_index") == day).select(
        "customer_id", "article_id", "source_rank"
    )


def run_checksum(spark, args, phase: int) -> dict:
    """
    Reproduce the shipped table on one day, source by source.

    Returns the comparison. Raises on any mismatch, naming the source -- which is
    the entire reason this is decomposed rather than a single union assertion.
    """
    ref = json.loads(Path(args.expect_from).read_text())
    ref_ceiling = ref["ceiling"]

    day = args.ann_snapshot_day
    cohort = spark.read.parquet("artifacts/twotower/eval_customers").select(
        "customer_id"
    ).distinct()
    truth = B.truth_pairs(spark, "val_tune").join(
        F.broadcast(cohort), "customer_id", "inner"
    )
    n_pairs = truth.count()
    if n_pairs != args.expect_pairs:
        raise SystemExit(
            f"denominator is {n_pairs}, expected {args.expect_pairs} -- the "
            "cohort moved and nothing below would be comparable"
        )

    events = cohort.withColumn("day_index", F.lit(int(day)))
    ann, ann_meta = load_ann_snapshot(spark, args.ann_snapshot, day)
    pairs = spark.read.parquet(str(args.pairs)).filter(
        F.col("anchor_day") == int(day)
    )
    if pairs.limit(1).count() == 0:
        raise SystemExit(
            f"no covisit pairs for anchor {day} under {args.pairs}. The checksum "
            "day must be an anchor -- that is what the val_tune phasing is for. "
            "Run covisit_pairs_table first."
        )

    sources = build_sources(
        spark, events, pairs, phase=phase, cadence=args.cadence, ann=ann,
        n_covisit=args.n_covisit, covisit_lookback=args.covisit_lookback,
        covisit_max_basket=args.covisit_max_basket,
    )
    single = {k: _as_single_day(v, day) for k, v in sources.items()}
    names = tuple(n for n in SOURCE_ORDER if n in single)
    union = C.union_candidates(*(single[n] for n in names), source_names=names)
    got = C.recall_ceiling(union, truth, source_names=names)

    rows, failures = [], []
    for key in ("recall_ceiling", *(f"by_{n}" for n in names)):
        want, have = ref_ceiling[key], got[key]
        ok = abs(want - have) <= CHECKSUM_TOL
        rows.append((key, want, have, ok))
        if not ok:
            failures.append(key)

    print(f"\nCHECKSUM against {args.expect_from}  (day {day}, {n_pairs} pairs)")
    print(f"{'metric':<22}{'shipped':>12}{'got':>12}   ")
    for key, want, have, ok in rows:
        print(f"{key:<22}{want*100:>11.4f}%{have*100:>11.4f}%   {'ok' if ok else 'MISMATCH'}")

    result = {
        "day": int(day),
        "n_true_pairs": n_pairs,
        "reference": str(args.expect_from),
        "ann_snapshot": ann_meta,
        "metrics": {k: {"shipped": w, "got": g, "ok": o} for k, w, g, o in rows},
        "passed": not failures,
    }
    if failures:
        raise SystemExit(
            f"checksum FAILED on {failures} -- the per-day machinery does not "
            "reproduce the shipped table. Do not run the 692-day job."
        )
    print("CHECKSUM PASS")
    return result


def assert_snapshot_single_day(lo, hi, snapshot, snapshot_day) -> None:
    """
    A single-day snapshot may only ever serve a single-day run.

    Without this, `--ann-snapshot` on the 692-day job would stamp
    2020-08-12's retrieval onto every day in the range. That is the per-day-ANN
    leak in its purest form, and its symptom is every downstream number getting
    BETTER -- so nothing would flag it. The guard is the PIT rule applied to the
    adapter, not defensiveness about a flag.
    """
    if snapshot is None:
        return
    if (lo, hi) != (snapshot_day, snapshot_day):
        raise SystemExit(
            f"--ann-snapshot is a single-day fixture for day {snapshot_day}, but "
            f"the requested range is {lo}..{hi}. Stamping it across multiple days "
            "would score every one of them with that day's retrieval. Use the "
            "per-day ANN stage for a real run."
        )


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checksum-day", action="store_true",
                   help="reproduce the shipped table on the snapshot day and exit")
    p.add_argument("--lo-day", type=int, default=None)
    p.add_argument("--hi-day", type=int, default=None)
    p.add_argument("--cadence", type=int, default=CD.COVISIT_CADENCE_DAYS)
    p.add_argument("--n-covisit", type=int, default=60)
    p.add_argument("--covisit-lookback", type=int, default=90)
    p.add_argument("--covisit-max-basket", type=int, default=50)
    p.add_argument("--pairs", type=Path, default=None,
                   help="default: $MARKETRANK_TABLES/covisit_pairs")
    p.add_argument("--out", type=Path, default=None,
                   help="default: $MARKETRANK_TABLES/candidates")
    # THE SNAPSHOT ADAPTER. Named for what it is; guarded below.
    p.add_argument("--ann-snapshot", type=Path, default=None,
                   help="single-day ANN parquet, checksum fixture ONLY")
    p.add_argument("--ann-snapshot-day", type=int, default=None,
                   help="the day that snapshot belongs to; defaults to val_tune day 0")
    p.add_argument("--expect-from", type=Path, default=SHIPPED_CEILING)
    p.add_argument("--expect-pairs", type=int, default=70_715)
    p.add_argument("--driver-memory", default="48g")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    from marketrank.spark import get_spark

    a = parse_args(argv)
    spark = get_spark("build_candidates", driver_memory=a.driver_memory)
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")

    phase = CD.covisit_phase(spark)
    if a.ann_snapshot_day is None:
        a.ann_snapshot_day = phase
    if a.pairs is None:
        a.pairs = config.TABLES / "covisit_pairs"
    if a.out is None:
        a.out = config.TABLES / "candidates"

    if a.checksum_day:
        if a.ann_snapshot is None:
            raise SystemExit("--checksum-day needs --ann-snapshot")
        return run_checksum(spark, a, phase)

    lo = CD.WARM_UP_DAYS if a.lo_day is None else a.lo_day
    hi = (phase - 1) if a.hi_day is None else a.hi_day
    assert_snapshot_single_day(lo, hi, a.ann_snapshot, a.ann_snapshot_day)
    raise SystemExit(
        "the multi-day write path is not implemented yet -- C1 lands the "
        "checksum first, on purpose. Run with --checksum-day."
    )


if __name__ == "__main__":
    main()
