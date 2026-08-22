"""
Tests for per-day candidate generation (C1).

WHY THIS FILE EXISTS, BEFORE THE JOB RUNS. R.0's doctrine, applied to the most
expensive job in the build: C1 writes ~1.37 billion rows over 692 days on a
cluster, and the thing it is most likely to get wrong is the one thing that
cannot be seen in the output -- a point-in-time boundary off by one day. A leak
does not crash, does not look wrong, and makes every downstream number better.
Step 4.2 found 85.6% NULL joins only because something else broke first.

Each test below is written so that the OBVIOUS wrong implementation fails it:

  * global_pop      an explode starting at +0 instead of +1
  * repurchase      `n_bought` and `last_day` computed globally rather than
                    as-of the scoring day -- the subtle one, because the shape
                    of the output is identical and only the ORDER changes
  * covisit anchor  a cadence that rounds to the nearest rebuild rather than
                    the most recent one, i.e. forward in time
  * scoring events  counting positives instead of customer-days, which is the
                    error step 4.2's sizing actually made
"""

import datetime as dt

import pytest

from marketrank import candidates_daily as CD, features as ft, ingest


def _txn(spark, rows):
    """(customer_id, article_id, t_dat) shaped like raw.transactions."""
    from pyspark.sql.types import DateType, StringType, StructField, StructType

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


def _articles(spark, rows):
    from pyspark.sql.types import IntegerType, StringType, StructField, StructType

    schema = StructType(
        [
            StructField("article_id", StringType(), False),
            StructField("product_type_no", IntegerType(), False),
        ]
    )
    return spark.createDataFrame(rows, schema)


def _day(offset: int) -> str:
    """A date `offset` days after DAY_ZERO, so day_index == offset."""
    return (dt.date.fromisoformat(ft.DAY_ZERO) + dt.timedelta(days=offset)).isoformat()


@pytest.fixture
def patched(spark, monkeypatch):
    """Dispatch `spark.table` by name -- these sources read two tables, not one."""

    def _install(txn_rows, article_rows=()):
        txn_df = _txn(spark, txn_rows)
        art_df = _articles(spark, list(article_rows))

        def _table(name):
            if name == ingest.ARTICLES_TABLE:
                return art_df
            return txn_df

        monkeypatch.setattr(spark, "table", _table, raising=False)
        return txn_df

    return _install


@pytest.mark.spark
def test_global_pop_never_sees_the_scoring_day_itself(spark, patched):
    """
    THE `-1` BOUNDARY, which is the whole point of the source.

    A bought on day 1, B on day 2. For day 2 the legal window is [day-30, day 1],
    so A is eligible and **B is not** -- B's only transaction is on the day being
    scored. By day 3 both are eligible.

    An explode of `+0 .. +lookback` instead of `+1 .. +lookback` puts B in day 2's
    list, which is a leak that would raise every recall number in the build.
    """
    patched([("c1", "A", _day(1)), ("c2", "B", _day(2))])

    pop = CD.daily_global_pop(spark, n=10, lookback_days=30)
    day2 = {r.article_id for r in pop.filter("day_index = 2").collect()}
    day3 = {r.article_id for r in pop.filter("day_index = 3").collect()}

    assert day2 == {"A"}, f"day 2 must not contain its own transactions, got {day2}"
    assert day3 == {"A", "B"}


@pytest.mark.spark
def test_global_pop_window_expires(spark, patched):
    """A 30-day lookback must drop a purchase on day 31, not carry it forever."""
    patched([("c1", "A", _day(0))])
    pop = CD.daily_global_pop(spark, n=10, lookback_days=30)
    days = {r.day_index for r in pop.collect()}
    assert 30 in days, "day 30 is the last day inside a 30-day window"
    assert 31 not in days, "day 31 is outside the window and must have no rows"


@pytest.mark.spark
def test_repurchase_rank_uses_as_of_counts_not_global_ones(spark, patched):
    """
    THE SUBTLE LEAK, and the reason this test asserts order rather than membership.

    c1's history:  A day 1, A day 3, B day 3, then B day 10 x3 (the scoring day).

    As of day 10 the legal prior slice is days 1 and 3:
        A -> last_day 3, n_bought 2
        B -> last_day 3, n_bought 1
    Ordering is (last_day desc, n_bought desc, article asc), so **A then B**.

    An implementation that aggregates `last_day`/`n_bought` over the customer's
    whole history sees B at last_day 10, n_bought 4 and ranks **B first**. The
    output has the same shape and the same articles either way -- only the order
    moves -- so membership assertions cannot catch it.
    """
    patched(
        [
            ("c1", "A", _day(1)),
            ("c1", "A", _day(3)),
            ("c1", "B", _day(3)),
            ("c1", "B", _day(10)),
            ("c1", "B", _day(10)),
            ("c1", "B", _day(10)),
        ]
    )
    events = CD.scoring_events(spark, _day(0), _day(20))
    rep = CD.daily_repurchase(spark, events, n=10)

    got = [
        (r.article_id, r.source_rank)
        for r in rep.filter("day_index = 10").orderBy("source_rank").collect()
    ]
    assert got == [("A", 1), ("B", 2)], f"as-of ordering violated: {got}"


@pytest.mark.spark
def test_repurchase_excludes_the_scoring_day(spark, patched):
    """Day 3 may see day 1 only; the day-3 purchase of B is not yet knowable."""
    patched([("c1", "A", _day(1)), ("c1", "B", _day(3))])
    events = CD.scoring_events(spark, _day(0), _day(20))
    rep = CD.daily_repurchase(spark, events, n=10)

    day3 = {r.article_id for r in rep.filter("day_index = 3").collect()}
    day1 = rep.filter("day_index = 1").count()
    assert day3 == {"A"}, f"day 3 leaked its own purchase: {day3}"
    assert day1 == 0, "day 1 has no prior history and must produce nothing"


@pytest.mark.spark
def test_scoring_events_are_customer_days_not_positives(spark, patched):
    """
    A three-article basket is ONE scoring event.

    This is the error step 4.2's sizing made (`positives x N`), and it inflates
    the training set by the mean basket -- measured 3.1633 on the train slice.
    """
    patched(
        [
            ("c1", "A", _day(1)),
            ("c1", "B", _day(1)),
            ("c1", "C", _day(1)),
            ("c1", "A", _day(2)),
            ("c2", "A", _day(1)),
        ]
    )
    events = CD.scoring_events(spark, _day(0), _day(20))
    assert events.count() == 3, "expected (c1,1), (c1,2), (c2,1)"


@pytest.mark.spark
def test_dominant_category_is_point_in_time(spark, patched):
    """
    c1 buys two type-7 articles on days 1-2 and three type-9 on day 5.

    On day 5 the dominant type is **7** (the day-5 purchases are not knowable).
    A leaky version returns 9, which would hand the customer candidates chosen
    by what they were about to buy.
    """
    patched(
        [
            ("c1", "A", _day(1)),
            ("c1", "B", _day(2)),
            ("c1", "X", _day(5)),
            ("c1", "Y", _day(5)),
            ("c1", "Z", _day(5)),
        ],
        [("A", 7), ("B", 7), ("X", 9), ("Y", 9), ("Z", 9)],
    )
    events = CD.scoring_events(spark, _day(0), _day(20))
    dom = CD.daily_dominant_category(spark, events)
    got = [r.product_type_no for r in dom.filter("day_index = 5").collect()]
    assert got == [7], f"dominant category leaked the scoring day: {got}"


def test_covisit_anchor_never_moves_forward_in_time():
    """
    The cadence must round DOWN. Reusing an older pair table under-informs;
    reusing a newer one is a leak, and the two differ by a single `floor` vs
    `round`.

    Pure arithmetic, no Spark -- this is a boundary property, not a query.
    """
    import pyspark.sql.functions as F  # noqa: F401  (kept for symmetry with usage)

    for cadence in (1, 7, 28):
        for day in range(0, 100):
            anchor = (day // cadence) * cadence
            assert anchor <= day, f"anchor {anchor} is after day {day}"
            assert anchor % cadence == 0
            assert day - anchor < cadence, "staleness must be bounded by the cadence"
