"""
The point-in-time feature pipeline.

**Point-in-time correctness in this pipeline is enforced by the window frame,
not by the join.** Every rolling feature is computed over `[d - w, d - 1]` with

    Window.partitionBy(...).orderBy("day_index").rangeBetween(-w, -1)

so the row for day `d` is already stamped "as of end of day d-1". Attaching it
to an event on day `d` is then a plain equi-join on `(entity, day_index)` -- no
as-of join machinery, no interval logic, no correlated subquery. The expensive
semantics live in the frame; what is left is cheap.

Two details carry the whole design:

* **The upper bound is -1.** Change it to 0 and every feature includes the
  event's own day. The ranker's AUC goes up and nothing else reports a problem.
  `tests/test_pit.py` exists for that one character.
* **`rangeBetween`, not `rowsBetween`.** Range frames compare *values* of the
  ordering expression, so `rangeBetween(-7, -1)` means "day_index in [d-7, d-1]"
  regardless of gaps. `rowsBetween(-7, -1)` means "the previous 7 rows", which
  for a customer who shopped 7 times in two years is a two-year window --
  silently wrong, and the features look plausible.

`t_dat` is a DATE with no time component, so two events on the same day cannot
be ordered and *any* same-day inclusion would leak an unknowable amount of
future. Excluding the whole day is the only defensible rule; it costs real
signal and that tradeoff is stated in the README.
"""

from __future__ import annotations

from datetime import date, timedelta

from pyspark.sql import Column, DataFrame, SparkSession, Window, functions as F

from marketrank import config

# Day 0 of the dataset. One integer per calendar day, and the thing the window
# frame orders on.
DAY_ZERO = "2018-09-20"

WINDOWS = (7, 30, 90)
MAX_WINDOW = max(WINDOWS)

# Money is DECIMAL, not DOUBLE, and this is not a style choice -- it is what
# makes the step 2.4 backfill checkpoint pass.
#
# Floating-point addition is not associative, and the summation order inside a
# hash aggregate depends on how the input was partitioned. A full build reads
# 734 days and a backfill reads 120, so the two runs split the input
# differently, sum the same prices in a different order, and produce `spend`
# values that differ in the last bit. Measured on this dataset: identical
# integer counts, but 23,758 of 469,829 customer-day rows differed in
# cust_spend_90d, max |delta| = 1.78e-15. Nothing is *wrong* with either number
# and every downstream model would be unaffected -- but "recompute a window and
# diff it against the full run" then fails forever, and the check that was
# supposed to catch the truncation trap gets disabled as noisy.
#
# Decimal addition is exact and therefore order-independent. price has 9,857
# distinct values in [1.69e-05, 0.5915] and all of them survive DECIMAL(10,8)
# without collision (step 1.4), so 10 decimal places is comfortably lossless
# for this column.
PRICE_TYPE = "decimal(12,10)"


def day_index(col: str | Column) -> Column:
    """Calendar day as an integer offset from DAY_ZERO."""
    if isinstance(col, str):
        col = F.col(col)
    return F.datediff(col, F.lit(DAY_ZERO).cast("date")).cast("int")


def with_day_index(df: DataFrame, date_col: str = "t_dat") -> DataFrame:
    """Attach `feature_date` and `day_index`, and carry BOTH downstream.

    `day_index` is what the window frame orders on; `feature_date` is what the
    Iceberg `days(...)` partition spec needs and what makes the table readable.
    Dropping the date here and reconstructing it at the write layer is a
    `date_add` on every row plus a chance to get the epoch wrong.
    """
    return (
        df.withColumn("feature_date", F.col(date_col).cast("date"))
        .withColumn("day_index", day_index("feature_date"))
    )


# --------------------------------------------------------------------------
# Step 2.2 -- the daily aggregate layer
#
# Rolling windows recomputed per event over 32M raw rows is the shape that
# explodes. Rolling windows over ~pre-aggregated daily rows is cheap and exact.
# The measures below describe the events ON that day; the rolling layer turns
# them into strictly-prior features.
# --------------------------------------------------------------------------

CUSTOMER_MEASURES = ("n_txn", "n_articles", "spend")
ARTICLE_MEASURES = ("n_txn", "n_customers", "spend")
CROSS_MEASURES = ("n_txn",)


def daily_customer_agg(txn: DataFrame) -> DataFrame:
    """(customer_id, feature_date, day_index, n_txn, n_articles, spend, avg_price)."""
    return (
        with_day_index(txn)
        .groupBy("customer_id", "feature_date", "day_index")
        .agg(
            F.count(F.lit(1)).alias("n_txn"),
            # Distinct articles bought THAT DAY. The rolling layer sums these,
            # so cust_n_articles_30d counts (day, article) pairs rather than
            # distinct articles over 30 days -- an exact distinct count over a
            # range frame is not expressible as a window aggregate, and the
            # cheap alternative is close enough for a count feature. Named so
            # the difference is visible rather than assumed.
            F.countDistinct("article_id").alias("n_articles"),
            F.sum(F.col("price").cast(PRICE_TYPE)).alias("spend"),
            F.avg(F.col("price").cast(PRICE_TYPE)).alias("avg_price"),
        )
    )


def daily_article_agg(txn: DataFrame) -> DataFrame:
    """(article_id, feature_date, day_index, n_txn, n_customers, spend, avg_price)."""
    return (
        with_day_index(txn)
        .groupBy("article_id", "feature_date", "day_index")
        .agg(
            F.count(F.lit(1)).alias("n_txn"),
            F.countDistinct("customer_id").alias("n_customers"),
            F.sum(F.col("price").cast(PRICE_TYPE)).alias("spend"),
            F.avg(F.col("price").cast(PRICE_TYPE)).alias("avg_price"),
        )
    )


def daily_cross_agg(txn: DataFrame, articles: DataFrame) -> DataFrame:
    """
    (customer_id, product_type_no, feature_date, day_index, n_txn).

    NOTE: this join is NOT point-in-time. `articles` is a current-state snapshot
    with no history and no valid-from/valid-to, so a 2018 event is labelled with
    a 2020 product type. Unavoidable with this dataset -- see README limitations.
    """
    cats = articles.select("article_id", "product_type_no")
    return (
        with_day_index(txn)
        .join(F.broadcast(cats), on="article_id", how="inner")
        .groupBy("customer_id", "product_type_no", "feature_date", "day_index")
        .agg(F.count(F.lit(1)).alias("n_txn"))
    )


# --------------------------------------------------------------------------
# Step 2.3 -- the rolling windows
# --------------------------------------------------------------------------


def rolling_features(
    daily_df: DataFrame,
    partition_cols: list[str],
    measures: tuple[str, ...],
    windows: tuple[int, ...] = WINDOWS,
    prefix: str = "",
    spine: DataFrame | None = None,
) -> DataFrame:
    """
    Turn a daily aggregate into strictly-prior rolling features.

    Returns `partition_cols + [feature_date, day_index] + <prefix>_<measure>_<w>d`
    and NOTHING same-day: the daily measures themselves are dropped, because a
    feature row for day `d` that carried day `d`'s own counts would be a leak
    wearing the pipeline's own naming convention.

    `spine` is every (entity, day) pair features are needed FOR, whether or not
    the entity transacted that day. Window functions only emit output for rows
    that exist, so without a spine, features exist only where a purchase
    happened -- which is its own, extremely flattering, kind of leak, because at
    scoring time candidates are evaluated on days the customer bought nothing.
    The pattern is: union the spine into the daily aggregate with zero-filled
    measures, window over the union, then filter back to the spine.
    """
    keys = partition_cols + ["feature_date", "day_index"]

    if spine is None:
        base = daily_df
    else:
        spine_keys = spine.select(*keys).distinct()
        base = (
            spine_keys.unionByName(daily_df.select(*keys))
            .distinct()
            .join(daily_df, on=keys, how="left")
        )
        for m in measures:
            base = base.withColumn(
                m, F.coalesce(F.col(m), F.lit(0).cast(daily_df.schema[m].dataType))
            )

    out = base.select(*keys, *measures)
    generated: list[str] = []
    for w in windows:
        frame = (
            Window.partitionBy(*partition_cols)
            .orderBy("day_index")
            .rangeBetween(-w, -1)  # -1, NOT 0. See the module docstring.
        )
        for m in measures:
            name = f"{prefix}{m}_{w}d"
            summed = F.sum(m).over(frame)
            out = out.withColumn(
                name, F.coalesce(summed, F.lit(0).cast(out.schema[m].dataType))
            )
            generated.append(name)
        # A derived measure that is only correct if computed from the two sums
        # over the SAME frame -- averaging an average would weight days equally
        # instead of transactions.
        if "spend" in measures and "n_txn" in measures:
            name = f"{prefix}avg_price_{w}d"
            out = out.withColumn(
                name,
                F.col(f"{prefix}spend_{w}d")
                / F.nullif(F.col(f"{prefix}n_txn_{w}d"), F.lit(0)),
            )
            generated.append(name)

    out = out.select(*keys, *generated)

    if spine is not None:
        out = out.join(spine.select(*keys).distinct(), on=keys, how="inner")
    return out


def customer_features(
    txn: DataFrame,
    windows: tuple[int, ...] = WINDOWS,
    spine: DataFrame | None = None,
) -> DataFrame:
    return rolling_features(
        daily_customer_agg(txn),
        ["customer_id"],
        CUSTOMER_MEASURES,
        windows=windows,
        prefix="cust_",
        spine=spine,
    )


def article_features(
    txn: DataFrame,
    windows: tuple[int, ...] = WINDOWS,
    spine: DataFrame | None = None,
) -> DataFrame:
    return rolling_features(
        daily_article_agg(txn),
        ["article_id"],
        ARTICLE_MEASURES,
        windows=windows,
        prefix="art_",
        spine=spine,
    )


def cross_features(
    txn: DataFrame,
    articles: DataFrame,
    windows: tuple[int, ...] = WINDOWS,
    spine: DataFrame | None = None,
) -> DataFrame:
    return rolling_features(
        daily_cross_agg(txn, articles),
        ["customer_id", "product_type_no"],
        CROSS_MEASURES,
        windows=windows,
        prefix="cross_",
        spine=spine,
    )


# --------------------------------------------------------------------------
# Step 2.4 -- persist and backfill
# --------------------------------------------------------------------------

FEATURE_NAMESPACE = f"{config.CATALOG}.features"
CUSTOMER_FEATURE_TABLE = f"{FEATURE_NAMESPACE}.feature_customer_daily"
ARTICLE_FEATURE_TABLE = f"{FEATURE_NAMESPACE}.feature_article_daily"
CROSS_FEATURE_TABLE = f"{FEATURE_NAMESPACE}.feature_cross_daily"


def _write_partitioned(df: DataFrame, table: str, spark: SparkSession) -> None:
    """Partition-overwrite `table`, creating it partitioned by day on first use."""
    exists = spark.catalog.tableExists(table)
    writer = df.writeTo(table)
    if exists:
        writer.overwritePartitions()
    else:
        writer.partitionedBy(F.days("feature_date")).createOrReplace()


def customer_day_spine(
    spark: SparkSession, customers: DataFrame, start: str, end: str
) -> DataFrame:
    """
    Every (customer, day) pair in [start, end] for the given customers.

    Window functions only emit output for rows that EXIST, so without this the
    feature tables contain a row for (customer, day) only where that customer
    transacted -- and week 4 needs features for exactly the days they did not,
    because candidates are scored on days nothing was bought.

    Measured consequence of omitting it, on this build's own candidate table:
    only 2,818 of 20,000 evaluation customers had a customer-feature row on
    2020-08-12, so **85.6% of candidate rows joined to NULL customer features**.
    See BUILD_NOTES step 4.2.
    """
    days = spark.sql(
        f"SELECT explode(sequence(date'{start}', date'{end}', interval 1 day))"
        " AS feature_date"
    ).withColumn("day_index", day_index("feature_date"))
    return customers.select("customer_id").distinct().crossJoin(F.broadcast(days))


def feature_coverage(
    feature_df: DataFrame,
    entities: DataFrame,
    key_col: str,
    day_index_value: int,
) -> dict:
    """
    What fraction of `entities` actually has a feature row on a given day?

    This is the audit that turns the spine bug from a silent one into a loud
    one, and it is deliberately phrased over the FEATURE TABLE rather than over
    a model's input tensor. The reason is the thing that made the bug survive
    week 3: `retrieval.dataset.customer_context` left-joins the features and
    then wraps every numeric in `log1p(coalesce(col, 0.0))`, so a customer with
    no feature row arrives at the tower as a row of honest-looking zeros, not as
    a null. There was never a null to audit downstream -- week 4 only caught it
    because the candidate join does not coalesce.

    So: measure coverage where the rows are, before anything fills them in.
    Measured consequence of skipping this in week 3 -- 2,818 of 20,000 eval
    customers had a row (14.09%), and the tower scored the other 85.91% from
    all-zero rolling features (BUILD_NOTES steps 4.2 and R.1).
    """
    ent = entities.select(key_col).distinct()
    have = (
        feature_df.filter(F.col("day_index") == F.lit(day_index_value))
        .select(key_col)
        .distinct()
    )
    n_entities = ent.count()
    n_covered = ent.join(have, key_col, "inner").count()
    return {
        "day_index": day_index_value,
        "n_entities": n_entities,
        "n_covered": n_covered,
        "coverage": (n_covered / n_entities) if n_entities else float("nan"),
    }


def build_features(
    spark: SparkSession,
    start: str | None = None,
    end: str | None = None,
    windows: tuple[int, ...] = WINDOWS,
    customer_spine: DataFrame | None = None,
    article_spine: DataFrame | None = None,
) -> dict[str, int]:
    """
    Build (or backfill) the three daily feature tables for feature days
    ``[start, end]``.

    THE CONTRACT IS AN ASYMMETRY, and it is the whole function:

        READ  source days [start - MAX_WINDOW, end]
        WRITE feature days [start, end]

    Filtering the daily aggregates to ``[start, end]`` *before* windowing is the
    natural way to write this, it looks like an obvious optimisation, and it is
    the **backfill-truncation trap**: every recomputed window is then truncated
    at the left edge, so a backfilled day near ``start`` gets a 90-day feature
    computed from three days of data. No error, no null, just a number that is
    quietly wrong in a table that already passed its leakage tests.

    Related contract, from the other direction: if source day ``D`` changes, day
    ``D``'s own feature row is NOT invalid -- by the ``rangeBetween(-w, -1)``
    rule it never saw day ``D``. The invalid range is ``[D + 1, D + 90]``.

    One function does the full build and the backfill. A separate backfill
    script is the anti-pattern; "recompute an arbitrary past window without a
    bespoke script" means the parameter, not the script.
    """
    from marketrank import ingest

    max_window = max(windows)
    read_start = None
    if start is not None:
        read_start = (
            date.fromisoformat(start) - timedelta(days=max_window)
        ).isoformat()

    txn = spark.table(ingest.TRANSACTIONS_TABLE)
    if read_start is not None:
        txn = txn.filter(F.col("t_dat") >= F.lit(read_start).cast("date"))
    if end is not None:
        txn = txn.filter(F.col("t_dat") <= F.lit(end).cast("date"))
    articles = spark.table(ingest.ARTICLES_TABLE)

    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {FEATURE_NAMESPACE}")

    def clip(df: DataFrame) -> DataFrame:
        if start is not None:
            df = df.filter(F.col("feature_date") >= F.lit(start).cast("date"))
        if end is not None:
            df = df.filter(F.col("feature_date") <= F.lit(end).cast("date"))
        return df

    written: dict[str, int] = {}
    for table, df in (
        (
            CUSTOMER_FEATURE_TABLE,
            customer_features(txn, windows=windows, spine=customer_spine),
        ),
        (
            ARTICLE_FEATURE_TABLE,
            article_features(txn, windows=windows, spine=article_spine),
        ),
        (CROSS_FEATURE_TABLE, cross_features(txn, articles, windows=windows)),
    ):
        out = clip(df)
        _write_partitioned(out, table, spark)
        written[table] = spark.table(table).count()
    return written


if __name__ == "__main__":  # `python -m marketrank.features [start end]`
    import sys
    import time

    from marketrank.spark import get_spark

    _start = sys.argv[1] if len(sys.argv) > 1 else None
    _end = sys.argv[2] if len(sys.argv) > 2 else None
    _spark = get_spark("build_features")
    _t = time.time()
    _res = build_features(_spark, start=_start, end=_end)
    print("BUILD_SECONDS %.1f" % (time.time() - _t))
    for _t_name, _n in _res.items():
        print("TABLE_ROWS", _t_name, _n)
    _spark.stop()
