from pyspark.sql import DataFrame, SparkSession


def snapshot_ids(spark: SparkSession, table: str) -> list[int]:
    """All snapshot ids for a table, oldest first."""
    rows = spark.sql(
        f"SELECT snapshot_id FROM {table}.snapshots ORDER BY committed_at"
    ).collect()
    return [r.snapshot_id for r in rows]


def read_snapshot(spark: SparkSession, table: str, snapshot_id: int) -> DataFrame:
    """Read the table as it was at a given snapshot."""
    return (
        spark.read
        .format("iceberg")
        .option("snapshot-id", snapshot_id)
        .load(table)
    )


def assert_identical(a: DataFrame, b: DataFrame, ignore_cols=("_ingested_at",)) -> None:
    """Fail unless two DataFrames hold the same rows, ignoring pipeline metadata."""
    a = a.drop(*ignore_cols)
    b = b.drop(*ignore_cols)

    only_in_a = a.exceptAll(b).count()
    only_in_b = b.exceptAll(a).count()

    print(f"only in a: {only_in_a}")
    print(f"only in b: {only_in_b}")

    if only_in_a or only_in_b:
        raise AssertionError("frames differ — the load is NOT idempotent")
    print("IDENTICAL — the load is idempotent")