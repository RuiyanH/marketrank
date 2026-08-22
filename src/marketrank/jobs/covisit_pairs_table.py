"""
Materialise the co-visitation pair tables, one partition per anchor day -- C1.

    python -m marketrank.jobs.covisit_pairs_table --lo-day 90 --hi-day 692

**The first real reader of `config.TABLES`.** A4 introduced that variable and
flagged it as plumbed-and-inert; this closes the flag. On misha it resolves to
scratch, which is where an object of this size belongs -- `artifacts/` derives
from `PROJECT_ROOT`, which is `$HOME` with ~19 GiB free.

**Why materialise at all.** `daily_covisit` needs a pair table per anchor, and
each one is a 90-day self-join -- the most expensive object in the job. Deriving
them inline would make the 692-day run a single un-restartable stage that
recomputes everything after any failure. Written per anchor, the job restarts
where it stopped, and `build_candidates` reads one partitioned directory.

RESTART CORRECTNESS, in three parts, because "already there" is not one question:

1. **Did the write commit?** Spark's file output committer drops `_SUCCESS`
   after all parts land, so a killed job leaves a directory WITHOUT one. That
   distinguishes committed from half-written with no new machinery -- do not
   invent a second marker meaning the same thing.
2. **Is it THIS configuration's write?** `_SUCCESS` cannot answer that. An
   anchor written by an earlier run at 60/50 bounds has a perfectly valid one.
   So each anchor also carries a `meta.json` with its derivation args, written
   AFTER `_SUCCESS`. A partition is skipped only when the meta exists *and* its
   content-affecting args match this run. On mismatch the job **aborts and
   lists** the offending anchors rather than overwriting: silently replacing
   valid-but-different data is the same class of mistake as training on a
   truncated table, just quieter. `--force` is the deliberate escape.
3. **Is the run-level view consistent?** It is DERIVED from the per-anchor metas
   at the end, never maintained incrementally. That dissolves the ordering
   hazard in both directions -- no "sidecar says complete, job died before it
   flushed", and no reverse. The per-anchor meta is the source of truth; a
   restart recomputes the run file from disk, so it cannot disagree with what is
   actually on it.

**Per-anchor paths, per-anchor commits.** Not `partitionBy("anchor_day")` with
dynamic partition overwrite on one parent path: a job killed mid-commit under
that mode is exactly the failure being designed against. Each anchor is its own
`mode("overwrite")` write to its own directory, which Spark clears first.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from marketrank import candidates_daily as CD, config, covisit, splits

# Args that change the CONTENT of a given anchor's pair table. `cadence` and
# `phase` are deliberately NOT here: they decide WHICH anchors exist, not what
# an anchor at day D contains -- that depends only on D and these four. An
# anchor carried over from a run at a different cadence is therefore still
# valid data, and recording them at run level is enough.
CONTENT_ARGS = ("lookback_days", "max_basket", "top_k", "window_days")


def _meta_dir(out: Path) -> Path:
    """`_meta` because Spark's file listing skips `_`-prefixed entries.

    A stray `meta.json` beside the `anchor_day=N` directories would be picked up
    by partition discovery and fail the read as non-parquet.
    """
    return out / "_meta"


def anchor_path(out: Path, anchor: int) -> Path:
    """Hive-style, so reading the parent recovers `anchor_day` as a column."""
    return out / f"anchor_day={int(anchor)}"


def anchor_meta_path(out: Path, anchor: int) -> Path:
    return _meta_dir(out) / f"anchor_day={int(anchor)}.json"


def read_anchor_meta(out: Path, anchor: int) -> dict | None:
    p = anchor_meta_path(out, anchor)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        # A meta truncated by a kill is not a valid skip signal.
        return None


def anchor_state(out: Path, anchor: int, args: dict) -> str:
    """`missing` | `partial` | `mismatch` | `ok` -- the three questions above."""
    path = anchor_path(out, anchor)
    if not path.exists():
        return "missing"
    if not (path / "_SUCCESS").exists():
        return "partial"
    meta = read_anchor_meta(out, anchor)
    if meta is None:
        return "partial"
    if any(meta.get("args", {}).get(k) != args[k] for k in CONTENT_ARGS):
        return "mismatch"
    return "ok"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lo-day", type=int, default=None,
                   help=f"first scoring day; default WARM_UP_DAYS={CD.WARM_UP_DAYS}")
    p.add_argument("--hi-day", type=int, default=None,
                   help="last scoring day; default val_tune's first day (the checksum day)")
    p.add_argument("--cadence", type=int, default=CD.COVISIT_CADENCE_DAYS)
    p.add_argument("--lookback-days", type=int, default=90)
    p.add_argument("--max-basket", type=int, default=50)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--window-days", type=int, default=covisit.WINDOW_DAYS)
    p.add_argument("--out", type=Path, default=None,
                   help="default: $MARKETRANK_TABLES/covisit_pairs")
    p.add_argument("--force", action="store_true",
                   help="overwrite anchors whose recorded args differ from this run")
    p.add_argument("--driver-memory", default="48g")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    from marketrank.spark import get_spark

    a = parse_args(argv)
    out = a.out or (config.TABLES / "covisit_pairs")
    spark = get_spark("covisit_pairs_table", driver_memory=a.driver_memory)
    # Total row count is unchanged by this, but the per-anchor self-join is
    # skewed by heavy articles and AQE splits those partitions.
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")

    phase = CD.covisit_phase(spark)
    lo = CD.WARM_UP_DAYS if a.lo_day is None else a.lo_day
    hi = phase if a.hi_day is None else a.hi_day
    anchors = CD.anchor_days_for(spark, lo, hi, a.cadence, phase)

    content = {k: getattr(a, k) for k in CONTENT_ARGS}
    print(f"PHASE {phase}  (val_tune day 0 = {splits.bounds('val_tune')[0]})")
    print(f"DAYS  {lo}..{hi}   CADENCE {a.cadence}   ANCHORS {len(anchors)}")
    print(f"OUT   {out}")

    states = {x: anchor_state(out, x, content) for x in anchors}
    mismatched = [x for x, st in states.items() if st == "mismatch"]
    if mismatched and not a.force:
        raise SystemExit(
            f"{len(mismatched)} anchor(s) exist with DIFFERENT derivation args: "
            f"{mismatched[:10]}{'...' if len(mismatched) > 10 else ''}\n"
            "These are valid data from another configuration, not garbage. "
            "Re-run with --force to overwrite, or point --out elsewhere."
        )

    # Everything not `ok` is rebuilt. Mismatches only survive to this line under
    # --force, because the check above aborts on them otherwise.
    todo = [x for x, st in states.items() if st != "ok"]
    skipped = len(anchors) - len(todo)
    print(f"STATE ok={skipped} todo={len(todo)} "
          f"(partial={sum(1 for s in states.values() if s == 'partial')}, "
          f"mismatch={len(mismatched)})")

    _meta_dir(out).mkdir(parents=True, exist_ok=True)
    for i, anchor in enumerate(todo, 1):
        t0 = time.time()
        pairs = CD.weekly_covisit_pairs(
            spark, [anchor],
            lookback_days=a.lookback_days, max_basket=a.max_basket,
            top_k=a.top_k, window_days=a.window_days,
        ).drop("anchor_day")  # recovered from the path on read

        path = anchor_path(out, anchor)
        pairs.write.mode("overwrite").parquet(str(path))
        n_rows = spark.read.parquet(str(path)).count()

        # AFTER the write commits, never before: this file is what a restart
        # trusts, so it must not exist for a partition that does not.
        anchor_meta_path(out, anchor).write_text(json.dumps({
            "anchor_day": int(anchor),
            "rows": int(n_rows),
            "args": content,
            "cadence": a.cadence,
            "phase": phase,
            "seconds": round(time.time() - t0, 1),
        }, indent=2))
        print(f"ANCHOR {anchor:>5}  rows {n_rows:>9}  "
              f"{time.time() - t0:6.1f}s  [{i}/{len(todo)}]")

    # RUN SIDECAR, derived from disk rather than accumulated in memory.
    metas = {x: read_anchor_meta(out, x) for x in anchors}
    missing = [x for x, m in metas.items() if m is None]
    run = {
        "day_range": [lo, hi],
        "warm_up_days": CD.WARM_UP_DAYS,
        "warm_up_reason": (
            "Days before the longest lookback (90) have a truncated window in "
            "every source -- 30-day popularity, as-of repurchase, 90-day "
            "covisit -- so their candidate sets are degenerate rather than "
            "wrong. Excluded by default; generating the complete grid is a "
            "flagged run, not the default."
        ),
        "cadence": a.cadence,
        "phase": phase,
        "phase_reason": (
            "Anchored to val_tune's first day, not DAY_ZERO: on the epoch grid "
            "the checksum day (692) floors to 686, six days stale, and could "
            "not reproduce the shipped covisit solo."
        ),
        "args": content,
        "anchors": [int(x) for x in anchors],
        "n_anchors": len(anchors),
        "total_rows": sum(m["rows"] for m in metas.values() if m),
        "incomplete_anchors": missing,
    }
    (_meta_dir(out) / "run.json").write_text(json.dumps(run, indent=2))
    print(f"WROTE {_meta_dir(out) / 'run.json'}")
    print(f"TOTAL anchors {len(anchors)}  rows {run['total_rows']}  "
          f"incomplete {len(missing)}")
    if missing:
        raise SystemExit(f"{len(missing)} anchor(s) never completed: {missing[:10]}")
    return run


if __name__ == "__main__":
    main()
