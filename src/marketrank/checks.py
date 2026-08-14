from pyspark.sql import DataFrame, SparkSession

# Columns whose names start with this prefix describe the PIPELINE, not the
# world: when a row landed, which file delivered it. Idempotency is a claim
# about business content -- re-running a load must not change what the table
# says about the world -- so the row comparison excludes them by default.
PIPELINE_PREFIX = "_"


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


def pipeline_columns(df: DataFrame) -> tuple[str, ...]:
    """The pipeline-metadata columns of a DataFrame, by the prefix rule."""
    return tuple(c for c in df.columns if c.startswith(PIPELINE_PREFIX))


def assert_identical(
    a: DataFrame,
    b: DataFrame,
    ignore_cols: tuple[str, ...] | None = None,
) -> None:
    """
    Fail unless two DataFrames hold exactly the same rows, duplicates included.

    ``ignore_cols=None`` (the default) drops every column whose name starts with
    ``_`` from both sides before comparing -- pipeline metadata, not business
    content. Pass an explicit tuple to override, and ``()`` to demand a
    whole-row match including metadata.

    The default is a *rule*, not a list: ``_ingested_at`` (step 1.3) and
    ``_source_file`` (step 9.1) are both covered without editing this function.
    """
    if ignore_cols is None:
        ignore_cols = tuple(set(pipeline_columns(a)) | set(pipeline_columns(b)))

    a_cmp = a.drop(*ignore_cols) if ignore_cols else a
    b_cmp = b.drop(*ignore_cols) if ignore_cols else b

    if sorted(a_cmp.columns) != sorted(b_cmp.columns):
        raise AssertionError(
            f"column sets differ: {sorted(a_cmp.columns)} vs {sorted(b_cmp.columns)}"
        )
    b_cmp = b_cmp.select(*a_cmp.columns)

    only_in_a = a_cmp.exceptAll(b_cmp).count()
    only_in_b = b_cmp.exceptAll(a_cmp).count()

    print(f"ignored (pipeline metadata): {sorted(ignore_cols)}")
    print(f"only in a: {only_in_a}")
    print(f"only in b: {only_in_b}")

    if only_in_a or only_in_b:
        raise AssertionError("rows differ -- the load is NOT idempotent")
    print("IDENTICAL -- the load is idempotent")
