"""
Restartable partitioned writes -- one implementation, used by every C1 job.

Both `covisit_pairs_table` (per anchor) and `build_candidates` (per chunk) need
the same thing: write a partition, mark it, and on restart decide whether what
is already there can be trusted. Writing that twice would give the build two
restart semantics that drift; this is the one.

"ALREADY THERE" IS THREE QUESTIONS, and each needs a different marker:

1. **Did the write commit?** Spark's file output committer drops `_SUCCESS` after
   all parts land, so a killed job leaves a directory without one. That is the
   commit marker, it already exists, and nothing here reimplements it.
2. **Is it THIS configuration's write?** `_SUCCESS` cannot say. A partition
   written under different bounds has a perfectly valid one. So each partition
   also carries a `meta.json` naming its derivation args, written AFTER the
   commit -- so the file a restart trusts cannot exist for a partition that does
   not.
3. **Is the meta itself trustworthy?** A kill mid-flush leaves truncated JSON.
   That reads as incomplete, not as a crash.

ON MISMATCH THE CALLER ABORTS RATHER THAN OVERWRITING. Silently replacing
valid-but-different data is the same class of mistake as training on a truncated
table, and quieter: nothing raises, the numbers just change.

RUN METADATA IS DERIVED FROM THE PARTITION METAS, never accumulated. That
removes the ordering hazard in both directions -- no "run file says complete,
job died before it flushed", and no reverse -- because a restart recomputes it
from what is actually on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

MISSING = "missing"
PARTIAL = "partial"
MISMATCH = "mismatch"
OK = "ok"


def meta_dir(out: Path) -> Path:
    """`_meta`, because Spark's file listing skips `_`-prefixed entries.

    A stray `.json` beside the `key=value` directories would otherwise be picked
    up by partition discovery and fail the read as non-parquet.
    """
    return out / "_meta"


def part_path(out: Path, key: str, value: int | str) -> Path:
    """Hive-style, so reading the parent recovers `key` as a column."""
    return out / f"{key}={value}"


def part_meta_path(out: Path, key: str, value: int | str) -> Path:
    return meta_dir(out) / f"{key}={value}.json"


def read_part_meta(out: Path, key: str, value: int | str) -> dict | None:
    p = part_meta_path(out, key, value)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        # Truncated by a kill mid-flush. Not a valid skip signal.
        return None


def write_part_meta(out: Path, key: str, value: int | str, meta: dict) -> Path:
    """Call AFTER the parquet write commits, never before."""
    meta_dir(out).mkdir(parents=True, exist_ok=True)
    p = part_meta_path(out, key, value)
    p.write_text(json.dumps(meta, indent=2))
    return p


def part_state(
    out: Path,
    key: str,
    value: int | str,
    args: dict,
    content_args: tuple[str, ...],
) -> str:
    """`missing` | `partial` | `mismatch` | `ok`.

    `content_args` are the args that change what the partition CONTAINS. Args
    that only decide which partitions exist -- cadence, chunk width -- must be
    left out, or tuning them forces a full rebuild for no changed row.
    """
    path = part_path(out, key, value)
    if not path.exists():
        return MISSING
    if not (path / "_SUCCESS").exists():
        return PARTIAL
    meta = read_part_meta(out, key, value)
    if meta is None:
        return PARTIAL
    recorded = meta.get("args", {})
    if any(recorded.get(k) != args[k] for k in content_args):
        return MISMATCH
    return OK


def plan(
    out: Path,
    key: str,
    values: list,
    args: dict,
    content_args: tuple[str, ...],
    force: bool = False,
) -> tuple[list, dict[str, str]]:
    """
    (todo, states). Raises on mismatch unless `force`.

    Everything not `ok` is rebuilt: `missing` and `partial` unconditionally, and
    `mismatch` only under `force`, because without it this raises first.
    """
    states = {v: part_state(out, key, v, args, content_args) for v in values}
    mismatched = [v for v, st in states.items() if st == MISMATCH]
    if mismatched and not force:
        shown = mismatched[:10]
        raise SystemExit(
            f"{len(mismatched)} {key} partition(s) exist with DIFFERENT "
            f"derivation args: {shown}{'...' if len(mismatched) > 10 else ''}\n"
            "That is valid data from another configuration, not garbage. "
            "Re-run with --force to overwrite, or point the output elsewhere."
        )
    return [v for v, st in states.items() if st != OK], states


def derive_run_meta(
    out: Path, key: str, values: list, extra: dict | None = None
) -> dict:
    """Aggregate the partition metas. The source of truth is the disk."""
    metas = {v: read_part_meta(out, key, v) for v in values}
    incomplete = [v for v, m in metas.items() if m is None]
    run = dict(extra or {})
    run.update({
        "partition_key": key,
        "partitions": list(values),
        "n_partitions": len(values),
        "total_rows": sum(m.get("rows", 0) for m in metas.values() if m),
        "incomplete": incomplete,
    })
    meta_dir(out).mkdir(parents=True, exist_ok=True)
    (meta_dir(out) / "run.json").write_text(json.dumps(run, indent=2))
    return run
