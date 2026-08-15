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

from pyspark.sql import Column, DataFrame, Window, functions as F

# Day 0 of the dataset. One integer per calendar day, and the thing the window
# frame orders on.
DAY_ZERO = "2018-09-20"

WINDOWS = (7, 30, 90)
MAX_WINDOW = max(WINDOWS)


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
            F.sum("price").alias("spend"),
            F.avg("price").alias("avg_price"),
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
            F.sum("price").alias("spend"),
            F.avg("price").alias("avg_price"),
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
            .fillna(0, subset=list(measures))
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
            out = out.withColumn(name, F.coalesce(F.sum(m).over(frame), F.lit(0)))
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
