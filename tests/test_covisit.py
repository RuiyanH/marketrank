"""
Tests for the co-visitation source (R.5).

WHY THIS FILE EXISTS. R.0's doctrine: week 3 shipped two silent bugs in
untested measurement-path code, and the fix was to test the path before
spending compute on it. `covisit.py` is ~190 lines of self-join, window and
decay logic that feeds `marginal_per_slot` -- the exact number R.6's keep/demote
rule reads -- and it had no tests at all. These two are the minimum: the decay
and seed-weight arithmetic computed by hand, and the PIT boundary.

Both were confirmed to FAIL against a deliberately broken implementation before
being committed passing; see BUILD_NOTES step R.5b for the mutation table.
"""

import pytest

from marketrank import covisit


def _events(spark, rows):
    """(customer_id, article_id, t_dat) -> a DataFrame shaped like raw.transactions."""
    from pyspark.sql.types import DateType, StringType, StructField, StructType
    import datetime as dt

    schema = StructType(
        [
            StructField("customer_id", StringType(), False),
            StructField("article_id", StringType(), False),
            StructField("t_dat", DateType(), False),
        ]
    )
    return spark.createDataFrame(
        [(c, a, dt.date.fromisoformat(d)) for c, a, d in rows], schema
    )


@pytest.fixture
def patched_txn(spark, monkeypatch):
    """Point covisit's table read at a hand-built frame instead of the warehouse."""

    def _install(rows):
        df = _events(spark, rows)
        monkeypatch.setattr(
            covisit.ingest, "TRANSACTIONS_TABLE", "unused_but_referenced"
        )
        monkeypatch.setattr(spark, "table", lambda _name: df, raising=False)
        return df

    return _install


@pytest.mark.spark
def test_covisit_scores_are_the_hand_computed_decay_and_seed_weights(
    spark, patched_txn
):
    """
    Three customers, one seed, arithmetic checkable on paper.

    as_of = 2020-01-31, half life 30d. c1 and c2 both buy A and B on 2020-01-31
    minus 1 day (day 30 of January); c3 buys A and C on 2020-01-01.

      pair (A,B): two customers co-occur, newest day = 01-30, age = 1 day
                  w = 0.5 ** (1/30) = 0.977159  each  ->  score = 1.954318
      pair (A,C): one customer,        newest day = 01-01, age = 30 days
                  w = 0.5 ** (30/30) = 0.5      ->  score = 0.5

    The ordering B > C is what a naive undecayed count would also give, so the
    test asserts the SCORES, not the order -- an implementation that dropped the
    decay entirely would still rank correctly and pass an order-only test.
    """
    patched_txn(
        [
            ("c1", "A", "2020-01-30"), ("c1", "B", "2020-01-30"),
            ("c2", "A", "2020-01-30"), ("c2", "B", "2020-01-30"),
            ("c3", "A", "2020-01-01"), ("c3", "C", "2020-01-01"),
        ]
    )
    pairs = covisit.covisit_pairs(
        spark, "2020-01-31", lookback_days=60, window_days=7,
        decay_half_life=30.0, top_k=20, max_basket=50,
    )
    got = {
        (r.article_id, r.other_article_id): r.score
        for r in pairs.collect()
        if r.article_id == "A"
    }
    assert set(got) == {("A", "B"), ("A", "C")}
    assert got[("A", "B")] == pytest.approx(2 * 0.5 ** (1 / 30), rel=1e-9)
    assert got[("A", "C")] == pytest.approx(0.5, rel=1e-9)


@pytest.mark.spark
def test_covisit_is_point_in_time(spark, patched_txn):
    """
    An event ON `as_of` must not change anything -- the same `[.., d-1]` rule
    the feature pipeline enforces, in a second place.

    This is the covisit analogue of test_pit's truncation property: compute over
    data ending before `as_of`, then again with same-day events appended, and
    require the outputs to be identical. A `<=` where the code has `<` passes
    every ordering check and fails only this.
    """
    base = [
        ("c1", "A", "2020-01-30"), ("c1", "B", "2020-01-30"),
        ("c2", "A", "2020-01-29"), ("c2", "B", "2020-01-29"),
    ]
    future = [
        # Same-day (as_of) events, and a pair that exists ONLY on as_of.
        ("c1", "Z", "2020-01-31"), ("c3", "A", "2020-01-31"),
        ("c3", "Z", "2020-01-31"),
    ]

    def score_map(rows):
        patched_txn(rows)
        return {
            (r.article_id, r.other_article_id): round(r.score, 12)
            for r in covisit.covisit_pairs(
                spark, "2020-01-31", lookback_days=60, window_days=7,
                decay_half_life=30.0, top_k=20, max_basket=50,
            ).collect()
        }

    before = score_map(base)
    after = score_map(base + future)
    assert before == after, (
        "co-visitation saw events on as_of -- article Z should not appear and "
        "no score should move"
    )
    assert not any("Z" in k for k in after)
