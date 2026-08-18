"""
Item-item co-visitation -- step R.5's second ceiling raiser.

For article pairs bought by the same customer inside a short window, a
time-decayed co-occurrence count; a customer's candidates are then the top
co-visited articles of their own recent purchases.

WHY THIS SOURCE AND NOT MORE PERSONALISATION. Every retrieval number this build
has produced says the same thing: the short-horizon signal on this dataset is
**trend and sequence, not identity**. Exact-article repurchase has a 3.36%
ceiling against 64.34% at `product_type_no`; global popularity beats
dominant-category popularity at equal depth (3.168% vs 2.162%); the two-tower,
whose article side was five static categorical embeddings, lost to recent
popularity outright. Co-visitation is the cheapest structure that captures "what
gets bought with what, lately" without a model, and it is the workhorse of the
competition's strongest public solutions.

**PIT.** Everything is computed from `t_dat < as_of`. The co-occurrence counts,
the recency decay and the customer's recent-article list all read strictly prior
days, so the same `[d-w, d-1]` rule the feature pipeline enforces holds here --
in a second place, which is exactly where a leak sneaks back in.

**Cost control.** A self-join over 31.8M transaction rows is quadratic in each
customer's basket and is the shape that explodes. Three bounds keep it finite,
and each one is a modelling statement rather than a performance hack:

* `lookback_days` (default 60) -- co-occurrence older than two months is not
  what this source is for; popularity already covers the stable part.
* `window_days` (default 7) -- "bought together" means within a week, and the
  same-day case is the strongest version of it.
* `max_basket` (default 50) -- a customer with 400 distinct articles in the
  window contributes 160,000 pairs on their own and is not a co-visitation
  signal, they are a reseller.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window, functions as F

from marketrank import features as ft, ingest

LOOKBACK_DAYS = 60
WINDOW_DAYS = 7
DECAY_HALF_LIFE = 30.0
TOP_K_PER_ARTICLE = 20
MAX_BASKET = 50
RECENT_K = 10


def _recent_events(
    spark: SparkSession, as_of: str, lookback_days: int, max_basket: int
) -> DataFrame:
    """
    Distinct (customer, article, day_index) strictly before `as_of`, capped.

    The cap is on distinct (article, day) EVENTS, not distinct articles: an
    article bought on three days consumes three slots. That is the intended
    behaviour -- repeat purchases are signal here -- but it means `max_basket`
    bounds events rather than basket width, and the two differ for exactly the
    heavy customers the cap exists to bound.
    """
    txn = spark.table(ingest.TRANSACTIONS_TABLE).filter(
        (F.col("t_dat") < F.lit(as_of).cast("date"))
        & (F.col("t_dat") >= F.date_sub(F.lit(as_of).cast("date"), lookback_days))
    )
    ev = (
        ft.with_day_index(txn)
        .select("customer_id", "article_id", "day_index")
        .distinct()
    )
    # Keep each customer's most recent `max_basket` articles in the window.
    ranked = ev.withColumn(
        "r",
        F.row_number().over(
            Window.partitionBy("customer_id").orderBy(
                F.desc("day_index"), F.asc("article_id")
            )
        ),
    )
    return ranked.filter(F.col("r") <= max_basket).drop("r")


def covisit_pairs(
    spark: SparkSession,
    as_of: str,
    lookback_days: int = LOOKBACK_DAYS,
    window_days: int = WINDOW_DAYS,
    decay_half_life: float = DECAY_HALF_LIFE,
    top_k: int = TOP_K_PER_ARTICLE,
    max_basket: int = MAX_BASKET,
) -> DataFrame:
    """
    (article_id, other_article_id, score) -- the top `top_k` co-visited articles
    of each article, by time-decayed co-occurrence.

    The decay is on the RECENCY OF THE CO-OCCURRENCE relative to `as_of`, not on
    the gap between the two purchases: a pair bought together last week is
    stronger evidence of what is selling now than a pair bought together in
    June, and the gap is already bounded by `window_days`.
    """
    ev = _recent_events(spark, as_of, lookback_days, max_basket)
    as_of_day = (
        spark.sql(f"SELECT datediff(date'{as_of}', date'{ft.DAY_ZERO}') AS d")
        .collect()[0]
        .d
    )

    a = ev.selectExpr(
        "customer_id", "article_id as article_id", "day_index as day_a"
    )
    b = ev.selectExpr(
        "customer_id", "article_id as other_article_id", "day_index as day_b"
    )
    pairs = (
        a.join(b, "customer_id")
        .filter(F.col("article_id") != F.col("other_article_id"))
        .filter(F.abs(F.col("day_a") - F.col("day_b")) <= window_days)
    )
    newest = F.greatest(F.col("day_a"), F.col("day_b"))
    pairs = pairs.withColumn(
        "w", F.pow(F.lit(0.5), (F.lit(as_of_day) - newest) / F.lit(decay_half_life))
    )

    scored = pairs.groupBy("article_id", "other_article_id").agg(
        F.sum("w").alias("score")
    )
    ranked = scored.withColumn(
        "r",
        F.row_number().over(
            Window.partitionBy("article_id").orderBy(
                F.desc("score"), F.asc("other_article_id")
            )
        ),
    )
    return ranked.filter(F.col("r") <= top_k).drop("r")


def covisit_source(
    spark: SparkSession,
    as_of: str,
    customers: DataFrame | None = None,
    n: int = 40,
    recent_k: int = RECENT_K,
    lookback_days: int = LOOKBACK_DAYS,
    max_basket: int = MAX_BASKET,
    pairs: DataFrame | None = None,
    **kwargs,
) -> DataFrame:
    """
    (customer_id, article_id, source, source_rank) -- top-`n` co-visited
    articles of the customer's `recent_k` most recent prior purchases.

    A candidate's score sums the co-visitation weights over every seed article
    that produced it, so an article co-visited with three of the customer's
    recent purchases outranks one co-visited with a single purchase. Seeds are
    weighted by their own recency rank so the newest purchase leads.
    """
    from marketrank.candidates import SOURCE_COVISIT

    if pairs is None:
        pairs = covisit_pairs(
            spark, as_of, lookback_days=lookback_days, max_basket=max_basket, **kwargs
        )

    # Use the CALLER's cap, not the module constant. The seeds are then filtered
    # to `recent_k`, so with recent_k <= min(caps) the two are identical and this
    # was benign -- but it silently ignored the argument, and becomes a real bug
    # the moment recent_k > max_basket.
    ev = _recent_events(spark, as_of, lookback_days, max_basket)
    if customers is not None:
        ev = ev.join(customers.select("customer_id").distinct(), "customer_id", "inner")

    seeds = ev.withColumn(
        "r",
        F.row_number().over(
            Window.partitionBy("customer_id").orderBy(
                F.desc("day_index"), F.asc("article_id")
            )
        ),
    ).filter(F.col("r") <= recent_k)
    # Newest seed counts most; 1/r is a deliberate, documented choice rather
    # than a tuned one -- nothing here is tuned until a checkpoint passes.
    seeds = seeds.withColumn("seed_w", F.lit(1.0) / F.col("r"))

    hits = (
        seeds.join(pairs, "article_id", "inner")
        .groupBy("customer_id", F.col("other_article_id").alias("article_id"))
        .agg(F.sum(F.col("score") * F.col("seed_w")).alias("score"))
    )
    ranked = hits.withColumn(
        "source_rank",
        F.row_number().over(
            Window.partitionBy("customer_id").orderBy(
                F.desc("score"), F.asc("article_id")
            )
        ),
    ).filter(F.col("source_rank") <= n)

    return ranked.select(
        "customer_id",
        "article_id",
        F.lit(SOURCE_COVISIT).alias("source"),
        "source_rank",
    )
