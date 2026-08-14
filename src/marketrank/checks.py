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


def assert_identical(spark: SparkSession, table: str, snap_a: int, snap_b: int) -> None:
    """Fail unless two snapshots hold exactly the same rows, duplicates included."""
    a = read_snapshot(spark, table, snap_a)
    b = read_snapshot(spark, table, snap_b)

    only_in_a = a.exceptAll(b).count()
    only_in_b = b.exceptAll(a).count()

    print(f"only in {snap_a}: {only_in_a}")
    print(f"only in {snap_b}: {only_in_b}")

    if only_in_a or only_in_b:
        raise AssertionError("snapshots differ — the load is NOT idempotent")
    print("IDENTICAL — the load is idempotent")