"""
The pytest wrapper around checks.assert_identical.

This test DRIVES the idempotency property rather than observing it: it performs
two loads and compares the two resulting snapshots. An assertion about a
pipeline's behaviour belongs where the pipeline can be driven; an assertion
about a table's content belongs in dbt. (The tempting dbt version -- "the newest
two snapshots differ by zero rows" -- is a test of your recent shell history: it
correctly fails whenever the last thing you did was append a new day.)

It builds its own three-line CSV instead of reading the real extract, so it runs
in CI, where there is a JVM but no data.
"""

import os

import pytest

from marketrank import checks, config, ingest

pytestmark = pytest.mark.spark

CSV = """\
t_dat,customer_id,article_id,price,sales_channel_id
2019-06-01,cust_a,0663713001,0.0508,2
2019-06-01,cust_a,0663713001,0.0508,2
2019-06-01,cust_b,0108775015,0.0169,1
2019-06-02,cust_b,0663713001,0.0339,1
2019-06-02,cust_c,0108775015,0.0169,2
"""


@pytest.fixture
def tiny_extract(tmp_path, monkeypatch):
    """A stand-in for data/raw, holding a handful of transactions."""
    (tmp_path / "transactions_train.csv").write_text(CSV)
    monkeypatch.setattr(config, "DATA_RAW", tmp_path)
    return tmp_path


@pytest.fixture
def temp_table(spark):
    name = f"{config.CATALOG}.test_tmp.transactions_{os.getpid()}"
    spark.sql(f"DROP TABLE IF EXISTS {name} PURGE")
    ingest.create_transactions_table(spark, table=name)
    yield name
    spark.sql(f"DROP TABLE IF EXISTS {name} PURGE")


def test_reload_is_idempotent(spark, tiny_extract, temp_table):
    ingest.load_transactions(spark, start="2019-06-01", end="2019-06-01",
                             table=temp_table)
    snap_a = checks.snapshot_ids(spark, temp_table)[-1]

    ingest.load_transactions(spark, start="2019-06-01", end="2019-06-01",
                             table=temp_table)
    snap_b = checks.snapshot_ids(spark, temp_table)[-1]

    assert snap_a != snap_b, "the second load did not commit a snapshot"

    a = checks.read_snapshot(spark, temp_table, snap_a)
    b = checks.read_snapshot(spark, temp_table, snap_b)

    assert a.count() == 3
    checks.assert_identical(a, b)


def test_whole_row_diff_still_sees_the_metadata_column(
    spark, tiny_extract, temp_table
):
    """
    The exclusion has to be doing work, not hiding a real difference.

    If this test ever passes, `_ingested_at` stopped varying between runs and
    test_reload_is_idempotent is proving nothing.
    """
    ingest.load_transactions(spark, start="2019-06-02", end="2019-06-02",
                             table=temp_table)
    snap_a = checks.snapshot_ids(spark, temp_table)[-1]
    ingest.load_transactions(spark, start="2019-06-02", end="2019-06-02",
                             table=temp_table)
    snap_b = checks.snapshot_ids(spark, temp_table)[-1]

    a = checks.read_snapshot(spark, temp_table, snap_a)
    b = checks.read_snapshot(spark, temp_table, snap_b)

    with pytest.raises(AssertionError):
        checks.assert_identical(a, b, ignore_cols=())


def test_partition_overwrite_replaces_a_day_wholesale(
    spark, tiny_extract, temp_table
):
    """
    Step 1.2's Think-first answer, as an executable claim.

    Loading a *subset* of a day leaves only that subset. This is correct for
    "re-run the day from the source of truth" and wrong for "apply a delta",
    which is what week 9 has to solve.
    """
    ingest.load_transactions(spark, start="2019-06-01", end="2019-06-01",
                             table=temp_table)
    assert spark.table(temp_table).count() == 3

    corrections = tiny_extract / "transactions_train.csv"
    corrections.write_text(
        "t_dat,customer_id,article_id,price,sales_channel_id\n"
        "2019-06-01,cust_d,0108775015,0.0169,1\n"
    )
    ingest.load_transactions(spark, start="2019-06-01", end="2019-06-01",
                             table=temp_table)

    rows = spark.table(temp_table).collect()
    assert len(rows) == 1
    assert rows[0].customer_id == "cust_d"
