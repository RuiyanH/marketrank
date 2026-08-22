"""
Restart semantics for the per-anchor pair tables (C1).

No Spark: `anchor_state` is pure filesystem logic, and it decides whether a
99-anchor run reuses a partition or rebuilds it. Getting it wrong is expensive
in both directions -- rebuilding everything after each kill, or worse, skipping
a partition that is truncated or was built under different bounds.

"Already there" is three questions, and each of these tests pins one:

  * did the write COMMIT              -> `_SUCCESS`, which Spark writes itself
  * is it THIS configuration's write  -> per-anchor meta, written after
  * is the meta itself trustworthy    -> a kill can truncate JSON mid-flush
"""

import json

import pytest

from marketrank.jobs import covisit_pairs_table as P

ARGS = {"lookback_days": 90, "max_basket": 50, "top_k": 40, "window_days": 7}


def _partition(tmp_path, anchor=692, success=True, meta=None):
    d = P.anchor_path(tmp_path, anchor)
    d.mkdir(parents=True)
    (d / "part-00000.snappy.parquet").write_bytes(b"not really parquet")
    if success:
        (d / "_SUCCESS").write_text("")
    if meta is not None:
        P.anchor_meta_path(tmp_path, anchor).parent.mkdir(parents=True, exist_ok=True)
        P.anchor_meta_path(tmp_path, anchor).write_text(meta)
    return tmp_path


def test_absent_partition_is_missing(tmp_path):
    assert P.anchor_state(tmp_path, 692, ARGS) == "missing"


def test_directory_without_success_is_partial(tmp_path):
    """A killed job leaves parts behind and no `_SUCCESS`. Never reuse that."""
    _partition(tmp_path, success=False, meta=json.dumps({"args": ARGS}))
    assert P.anchor_state(tmp_path, 692, ARGS) == "partial"


def test_success_without_meta_is_partial(tmp_path):
    """
    The window between Spark committing and the meta landing.

    Treated as partial rather than ok: without the meta there is no evidence of
    WHICH configuration wrote it, and a rebuild is cheap next to trusting it.
    """
    _partition(tmp_path, success=True, meta=None)
    assert P.anchor_state(tmp_path, 692, ARGS) == "partial"


def test_truncated_meta_is_partial_not_a_crash(tmp_path):
    """A kill mid-flush leaves half a JSON document. It must not raise."""
    _partition(tmp_path, meta='{"anchor_day": 692, "args": {"lookba')
    assert P.anchor_state(tmp_path, 692, ARGS) == "partial"


def test_different_bounds_are_a_mismatch_not_a_skip(tmp_path):
    """
    THE ONE `_SUCCESS` CANNOT ANSWER.

    An anchor written at 60/50 has a perfectly valid `_SUCCESS` and a complete
    parquet directory. Reusing it would silently mix two configurations inside
    one candidate table -- valid data, wrong data, no error anywhere.
    """
    old = dict(ARGS, lookback_days=60)
    _partition(tmp_path, meta=json.dumps({"args": old}))
    assert P.anchor_state(tmp_path, 692, ARGS) == "mismatch"


def test_matching_args_are_ok(tmp_path):
    _partition(tmp_path, meta=json.dumps({"args": ARGS}))
    assert P.anchor_state(tmp_path, 692, ARGS) == "ok"


def test_cadence_and_phase_do_not_make_a_mismatch(tmp_path):
    """
    Cadence and phase decide WHICH anchors exist, not what an anchor contains.

    An anchor at day D built under a different cadence holds exactly the same
    pairs -- its content depends only on D and the four content args. Treating
    those as mismatches would force a full 99-anchor rebuild every time the
    cadence is tuned, for no change in a single row.
    """
    meta = json.dumps({"args": ARGS, "cadence": 28, "phase": 0})
    _partition(tmp_path, meta=meta)
    assert P.anchor_state(tmp_path, 692, ARGS) == "ok"


@pytest.mark.parametrize("field", sorted(ARGS))
def test_every_content_arg_is_actually_compared(tmp_path, field):
    """
    Guards the list itself. A content arg omitted from CONTENT_ARGS would let a
    genuinely different table pass as `ok`, and nothing else in the suite would
    notice.
    """
    stale = dict(ARGS)
    stale[field] = ARGS[field] + 1
    _partition(tmp_path, anchor=100, meta=json.dumps({"args": stale}))
    assert P.anchor_state(tmp_path, 100, ARGS) == "mismatch", (
        f"{field} is in CONTENT_ARGS but changing it did not register"
    )
