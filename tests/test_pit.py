"""
The point-in-time leakage tests. **This file is gate 1.**

Written before `marketrank.features` exists, and committed failing, so the git
history shows the order. A test written after the pipeline tends to test what
the pipeline does.

Both tests run against a hand-built DataFrame -- a dozen rows, three customers --
so they need a JVM but no data, and therefore run in CI.

Test 1 checks a boundary that was written deliberately, so it catches an
off-by-one. Test 2 checks a *property* -- the past is a function of the past --
and catches an entire class of bugs nobody anticipated: a global mean
imputation, an encoder fit on the full dataset, a percentile over all time, a
join that pulls a customer's lifetime total. None of those look like a window
bug, which is why they survive code review.
"""

import datetime as dt

import pytest

pytestmark = pytest.mark.spark

# Day 0 of the dataset. day_index is datediff(t_dat, DAY_ZERO).
DAY_ZERO = dt.date(2018, 9, 20)


def d(day_index: int) -> dt.date:
    return DAY_ZERO + dt.timedelta(days=day_index)


# Three customers, twelve rows.
#   c1 buys on days 10 and 11        -- the same-day / previous-day boundary
#   c2 buys twice on day 11          -- two events the same day, unorderable
#   c3 buys on days 1, 5, 40, 100    -- gaps wider than the 7d and 30d windows
ROWS = [
    ("c1", "a1", d(10), 0.10, 1),
    ("c1", "a2", d(11), 0.20, 1),
    ("c1", "a3", d(11), 0.30, 2),
    ("c2", "a1", d(11), 0.15, 1),
    ("c2", "a1", d(11), 0.15, 1),
    ("c2", "a4", d(12), 0.25, 2),
    ("c3", "a1", d(1), 0.05, 1),
    ("c3", "a2", d(5), 0.05, 1),
    ("c3", "a3", d(40), 0.50, 2),
    ("c3", "a4", d(100), 0.40, 1),
    ("c3", "a5", d(101), 0.45, 1),
    ("c1", "a5", d(120), 0.35, 2),
]

SCHEMA = "customer_id string, article_id string, t_dat date, price double, sales_channel_id int"


@pytest.fixture
def txn(spark):
    return spark.createDataFrame(ROWS, schema=SCHEMA)


def test_window_excludes_same_day(spark, txn):
    """
    c1 bought on day 10 and on day 11. The 7-day count attached to c1's day-11
    feature row must be exactly 1 -- the day-10 purchase and nothing else.

    If it is 2, the window's upper bound is 0 instead of -1 and every feature in
    the project includes the event's own day. That is a one-character bug, it
    raises the ranker's AUC, and nothing else in the system reports it.
    """
    from marketrank import features

    feats = features.customer_features(txn, windows=(7,))
    row = (
        feats.filter("customer_id = 'c1' and day_index = 11")
        .select("cust_n_txn_7d")
        .collect()
    )
    assert len(row) == 1, "expected exactly one feature row for (c1, day 11)"
    assert row[0].cust_n_txn_7d == 1, (
        "the 7-day count attached to c1's day-11 event must cover [4, 10] only. "
        "Got 2 => the frame's upper bound includes the event's own day."
    )


def test_future_events_do_not_change_past_features(spark, txn):
    """
    The strong one. Compute features over data truncated at day D, then over the
    full data, and assert every feature row for days <= D is identical, column
    for column.
    """
    from marketrank import features

    cutoff = 40

    truncated = txn.filter(f"t_dat <= date'{d(cutoff).isoformat()}'")

    full_feats = features.customer_features(txn).filter(f"day_index <= {cutoff}")
    trunc_feats = features.customer_features(truncated).filter(
        f"day_index <= {cutoff}"
    )

    assert sorted(full_feats.columns) == sorted(trunc_feats.columns)
    trunc_feats = trunc_feats.select(*full_feats.columns)

    assert full_feats.count() > 0, "the fixture produced no feature rows to compare"

    only_full = full_feats.exceptAll(trunc_feats).collect()
    only_trunc = trunc_feats.exceptAll(full_feats).collect()

    assert not only_full and not only_trunc, (
        "features for days <= cutoff changed when future events were added.\n"
        f"only in the full run:      {only_full[:5]}\n"
        f"only in the truncated run: {only_trunc[:5]}"
    )
