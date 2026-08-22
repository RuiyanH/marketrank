"""
Candidate generation with a PER-DAY `as_of` -- step C1, the ranker-scale set.

Everything in `candidates.py` takes a single `as_of` string and answers "what
would stage 1 retrieve for these customers on this one day". That was the right
object for R.5/R.6, where every number is measured on `val_tune`'s first day.
It is the wrong object for week 5: the ranker trains on 8,584,379 scoring events
spread over 692 days, and three of the five sources are TIME-VARYING. Scoring a
2019 event with 2020's popularity list is leakage, full stop.

**The scoring unit is (customer_id, day_index), not (customer_id).** A customer
who buys three articles on one day is ONE scoring event with ONE candidate set
and three positive labels -- measured mean basket 3.1633 over this slice. Getting
this wrong inflates the training set by that factor, which is exactly the error
step 4.2's sizing made (see Rider 1 in BUILD_NOTES).

**PIT.** Every source here reads strictly `< day`, i.e. `<= day - 1`. In window
terms that is `rangeBetween(-w, -1)`, and the `-1` is the whole point: an upper
bound of `0` would let a day's own transactions choose its candidates, and the
recall numbers would look wonderful and mean nothing. This is the second place
in the build that enforces that rule (the first is `features.py`), which is
exactly where a leak sneaks back in.

**Exactness requirement.** Restricting this job's output to the `val_tune`
cohort on a single day MUST reproduce the shipped 11.930% ceiling. That is the
checksum on the whole per-day machinery, and it is why each function below
reproduces its single-day sibling's ordering and tie-breaks precisely rather
than approximately -- including `repurchase`'s
`(last_dat desc, n_bought desc, article_id asc)`, where the first two keys are
themselves as-of-day quantities.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window, functions as F

from marketrank import candidates as C, covisit, features as ft, ingest
from marketrank.retrieval import baselines as B

# How often the co-visitation pair table is rebuilt. See `weekly_covisit_pairs`.
COVISIT_CADENCE_DAYS = 7


def _txn_days(spark: SparkSession) -> DataFrame:
    """Transactions with `day_index` attached, the ordering key for every window."""
    return ft.with_day_index(spark.table(ingest.TRANSACTIONS_TABLE))


def scoring_events(spark: SparkSession, lo: str, hi: str) -> DataFrame:
    """
    (customer_id, day_index) -- one row per customer-day with a purchase.

    THE denominator for everything downstream. Measured on the `train` slice:
    8,584,379 rows against 27,155,032 positives, mean basket 3.1633.
    """
    txn = _txn_days(spark).filter(
        (F.col("t_dat") >= F.lit(lo).cast("date"))
        & (F.col("t_dat") <= F.lit(hi).cast("date"))
    )
    return txn.select("customer_id", "day_index").distinct()


def _article_day_counts(spark: SparkSession, lookback_days: int) -> DataFrame:
    """
    (article_id, day_index, n) -- an article's transaction count in the window
    `[day - lookback, day - 1]`, for every day that window is non-empty.

    WHY EXPLODE RATHER THAN A ROLLING WINDOW. The obvious form is
    `sum(n).over(partitionBy(article).orderBy(day).rangeBetween(-w, -1))`, but a
    window only emits rows where the article ALREADY has one. An article bought
    on day 10 and not again until day 400 would get no row for days 11-40, even
    though its 30-day window is non-empty there -- so it would silently drop out
    of the top-N on precisely the days it should rank. Exploding each
    transaction-day forward across the days it influences produces the complete
    grid instead of a sparse one.

    The `+1 .. +lookback` bounds ARE the PIT rule: a transaction on day `t`
    informs day `t+1` at the earliest, never day `t` itself.
    """
    per_day = (
        _txn_days(spark)
        .groupBy("article_id", "day_index")
        .agg(F.count(F.lit(1)).alias("n"))
    )
    return (
        per_day.withColumn(
            "d",
            F.explode(
                F.sequence(
                    F.col("day_index") + F.lit(1),
                    F.col("day_index") + F.lit(lookback_days),
                )
            ),
        )
        .groupBy("article_id", F.col("d").alias("day_index"))
        .agg(F.sum("n").alias("n"))
    )


def daily_global_pop(
    spark: SparkSession,
    n: int = C.N_GLOBAL_POP,
    lookback_days: int = B.POPULARITY_LOOKBACK_DAYS,
) -> DataFrame:
    """(day_index, article_id, source_rank) -- top-`n` by recent volume, per day."""
    counts = _article_day_counts(spark, lookback_days)
    ranked = counts.withColumn(
        "source_rank",
        F.row_number().over(
            Window.partitionBy("day_index").orderBy(F.desc("n"), F.asc("article_id"))
        ),
    )
    return ranked.filter(F.col("source_rank") <= n).select(
        "day_index", "article_id", "source_rank"
    )


def daily_category_pop(
    spark: SparkSession,
    n: int = C.N_CATEGORY,
    lookback_days: int = B.POPULARITY_LOOKBACK_DAYS,
) -> DataFrame:
    """
    (day_index, product_type_no, article_id, source_rank).

    The article -> product_type_no mapping is a current-state snapshot (step
    2.0), deliberately: it is a dimension, not an event stream. Only the
    POPULARITY inside each category is point-in-time.
    """
    cats = spark.table(ingest.ARTICLES_TABLE).select("article_id", "product_type_no")
    counts = _article_day_counts(spark, lookback_days).join(
        F.broadcast(cats), "article_id", "inner"
    )
    ranked = counts.withColumn(
        "source_rank",
        F.row_number().over(
            Window.partitionBy("day_index", "product_type_no").orderBy(
                F.desc("n"), F.asc("article_id")
            )
        ),
    )
    return ranked.filter(F.col("source_rank") <= max(n, 1)).select(
        "day_index", "product_type_no", "article_id", "source_rank"
    )


def daily_dominant_category(spark: SparkSession, events: DataFrame) -> DataFrame:
    """
    (customer_id, day_index, product_type_no) -- the customer's most-bought
    product type strictly before each of their scoring days.

    Matches `candidates.category_popularity_source`'s tie-break exactly:
    `(count desc, product_type_no asc)`.
    """
    cats = spark.table(ingest.ARTICLES_TABLE).select("article_id", "product_type_no")
    prior = _txn_days(spark).join(F.broadcast(cats), "article_id", "inner")

    joined = (
        events.alias("e")
        .join(prior.alias("p"), "customer_id", "inner")
        .filter(F.col("p.day_index") < F.col("e.day_index"))
        .groupBy("customer_id", F.col("e.day_index").alias("day_index"), "product_type_no")
        .agg(F.count(F.lit(1)).alias("n"))
    )
    return (
        joined.withColumn(
            "r",
            F.row_number().over(
                Window.partitionBy("customer_id", "day_index").orderBy(
                    F.desc("n"), F.asc("product_type_no")
                )
            ),
        )
        .filter("r = 1")
        .select("customer_id", "day_index", "product_type_no")
    )


def daily_repurchase(
    spark: SparkSession, events: DataFrame, n: int = C.N_REPURCHASE
) -> DataFrame:
    """
    (customer_id, day_index, article_id, source_rank) -- the customer's own prior
    articles, most-recent-first, as of each scoring day.

    Reproduces `baselines.repurchase_ranks` per day, including its full ordering
    key `(last_dat desc, n_bought desc, article_id asc)`. Note that BOTH
    `last_dat` and `n_bought` are as-of-day quantities: they are recomputed from
    the prior slice for every scoring day, not carried from a global aggregate.
    Using a global `n_bought` would be a subtle leak -- a customer's total
    purchase count includes purchases that have not happened yet.

    NO LOOKBACK BOUND, matching the single-day version: repurchase reads the
    customer's whole prior history. That is what makes this the widest-fan-out
    join in the job, and it is deliberate rather than overlooked.
    """
    prior = _txn_days(spark).select("customer_id", "article_id", "day_index")
    joined = (
        events.alias("e")
        .join(prior.alias("p"), "customer_id", "inner")
        .filter(F.col("p.day_index") < F.col("e.day_index"))
        .groupBy(
            "customer_id",
            F.col("e.day_index").alias("day_index"),
            "article_id",
        )
        .agg(
            F.max(F.col("p.day_index")).alias("last_day"),
            F.count(F.lit(1)).alias("n_bought"),
        )
    )
    ranked = joined.withColumn(
        "source_rank",
        F.row_number().over(
            Window.partitionBy("customer_id", "day_index").orderBy(
                F.desc("last_day"), F.desc("n_bought"), F.asc("article_id")
            )
        ),
    )
    return ranked.filter(F.col("source_rank") <= n).select(
        "customer_id", "day_index", "article_id", "source_rank"
    )


def covisit_anchor_day(day_index_col, cadence_days: int = COVISIT_CADENCE_DAYS):
    """
    The most recent pair-table rebuild at or before `day_index`.

    THE CADENCE IS A COST DECISION WITH A SAFETY PROPERTY. Rebuilding the
    90-day pair self-join for all 692 days is ~692 derivations of the most
    expensive object in the job. Rebuilding weekly is 99, and reusing Monday's
    table for the rest of the week is **PIT-safe in the conservative direction**:
    Friday is scored with a table built from data strictly older than Monday, so
    it can UNDER-inform but can never leak. Staleness is at most 6 days against a
    90-day window.

    Making this finer is a pure cost increase; making it coarser trades fidelity
    the same way. It is not a correctness knob, which is why it is expressed as
    floor division rather than hidden in a config default.
    """
    return (F.floor(day_index_col / F.lit(cadence_days)) * F.lit(cadence_days)).cast(
        "int"
    )


def weekly_covisit_pairs(
    spark: SparkSession,
    anchor_days: list[int],
    lookback_days: int = 90,
    max_basket: int = 50,
    top_k: int = 40,
    window_days: int = covisit.WINDOW_DAYS,
) -> DataFrame:
    """
    (anchor_day, article_id, other_article_id, score) for each rebuild point.

    Defaults are the SHIPPED configuration (90/50, top_k 40) -- see R.6. This is
    the only function here that loops in Python rather than expressing itself as
    one Spark job, because `covisit_pairs` is a self-join whose cost is driven by
    the window, and unioning 99 independent self-joins into a single plan gives
    Spark no help while making the failure mode a single un-restartable stage.
    """
    frames = []
    for anchor in anchor_days:
        as_of = ft.day_index_to_date(anchor) if hasattr(ft, "day_index_to_date") else None
        if as_of is None:
            as_of = (
                spark.sql(
                    f"SELECT date_add(date'{ft.DAY_ZERO}', {int(anchor)}) AS d"
                )
                .collect()[0]
                .d.isoformat()
            )
        pairs = covisit.covisit_pairs(
            spark,
            as_of,
            lookback_days=lookback_days,
            window_days=window_days,
            top_k=top_k,
            max_basket=max_basket,
        )
        frames.append(pairs.withColumn("anchor_day", F.lit(int(anchor))))
    out = frames[0]
    for f in frames[1:]:
        out = out.unionByName(f)
    return out


def daily_covisit(
    spark: SparkSession,
    events: DataFrame,
    pairs: DataFrame,
    n: int = C.N_COVISIT,
    recent_k: int = covisit.RECENT_K,
    lookback_days: int = 90,
    max_basket: int = 50,
    cadence_days: int = COVISIT_CADENCE_DAYS,
) -> DataFrame:
    """
    (customer_id, day_index, article_id, source_rank).

    Seeds are the customer's `recent_k` most recent articles strictly before the
    scoring day, weighted `1/rank` exactly as the single-day version does; scores
    sum over seeds so an article co-visited with several recent purchases
    outranks one co-visited with a single purchase.
    """
    prior = _txn_days(spark).select("customer_id", "article_id", "day_index").distinct()
    seeds = (
        events.alias("e")
        .join(prior.alias("p"), "customer_id", "inner")
        .filter(F.col("p.day_index") < F.col("e.day_index"))
        .filter(
            F.col("p.day_index")
            >= F.col("e.day_index") - F.lit(lookback_days)
        )
        .select(
            "customer_id",
            F.col("e.day_index").alias("day_index"),
            "article_id",
            F.col("p.day_index").alias("seed_day"),
        )
    )
    # DO NOT DEDUPLICATE BY ARTICLE. `covisit._recent_events` ranks distinct
    # (customer, article, day) EVENTS, so an article bought on three days
    # occupies three seed slots and contributes 1/1 + 1/2 + 1/3 = 1.833 of seed
    # weight, not 1.0. Its docstring is explicit that this is intended --
    # "repeat purchases are signal here" -- and collapsing to one row per
    # article silently changes every covisit score, hence the top-n, hence the
    # solo coverage. Same ordering key as the single-day version, so `r` (and
    # therefore `seed_w`) match it row for row.
    #
    # One row_number covers both caps: `_recent_events` applies `max_basket`
    # and `covisit_source` then applies `recent_k` over the SAME ordering, so
    # ranks agree and filtering to min(recent_k, max_basket) is equivalent.
    seeds = (
        seeds.withColumn(
            "r",
            F.row_number().over(
                Window.partitionBy("customer_id", "day_index").orderBy(
                    F.desc("seed_day"), F.asc("article_id")
                )
            ),
        )
        .filter(F.col("r") <= min(recent_k, max_basket))
        .withColumn("seed_w", F.lit(1.0) / F.col("r"))
    )

    seeds = seeds.withColumn("anchor_day", covisit_anchor_day(F.col("day_index"), cadence_days))
    hits = (
        seeds.join(pairs, ["anchor_day", "article_id"], "inner")
        .groupBy(
            "customer_id",
            "day_index",
            F.col("other_article_id").alias("article_id"),
        )
        .agg(F.sum(F.col("score") * F.col("seed_w")).alias("score"))
    )
    ranked = hits.withColumn(
        "source_rank",
        F.row_number().over(
            Window.partitionBy("customer_id", "day_index").orderBy(
                F.desc("score"), F.asc("article_id")
            )
        ),
    )
    return ranked.filter(F.col("source_rank") <= n).select(
        "customer_id", "day_index", "article_id", "source_rank"
    )
