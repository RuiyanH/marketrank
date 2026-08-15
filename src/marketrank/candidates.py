"""
Candidate generation -- the union of three sources, each tagged with its origin.

TAG THE SOURCE. "Which source did this candidate come from" is a feature the
ranker uses, and it is also the retrieval diagnostic: if most of the ranker's
top-12 come from one source, the others are not contributing and you need to
know that before week 6's baseline comparison.

RECALL CEILING. The fraction of true purchases that appear anywhere in the
candidate set is the **hard ceiling on end-to-end recall** -- the ranker cannot
recover a purchase that stage 1 dropped. It is the honest measure of whether
stage 1 is doing its job, and it is what makes the two-stage argument concrete
rather than architectural.

Step 3.1 measured that exact-article repurchase covers only 3.36% of val_tune
purchases while category affinity covers 64%, so source 3 below is expected to
carry this set, not source 2. That expectation is checked, not assumed.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window, functions as F

from marketrank import ingest, splits
from marketrank.retrieval import baselines as B

SOURCE_ANN = "ann"
SOURCE_REPURCHASE = "repurchase"
SOURCE_CATEGORY = "category_pop"

N_REPURCHASE = 30
N_CATEGORY = 40
N_ANN = 50
CATEGORY_POOL = 200


def repurchase_source(spark: SparkSession, as_of: str, n: int = N_REPURCHASE) -> DataFrame:
    """(customer_id, article_id, source, source_rank) from the customer's own history."""
    return (
        B.repurchase_ranks(spark, as_of)
        .filter(F.col("rep_rank") <= n)
        .select(
            "customer_id",
            "article_id",
            F.lit(SOURCE_REPURCHASE).alias("source"),
            F.col("rep_rank").alias("source_rank"),
        )
    )


def category_popularity_source(
    spark: SparkSession,
    as_of: str,
    n: int = N_CATEGORY,
    lookback_days: int = B.POPULARITY_LOOKBACK_DAYS,
) -> DataFrame:
    """
    Top-N recently popular articles inside each customer's DOMINANT category.

    "Dominant category" is the product_type_no the customer has bought most often
    strictly before `as_of` -- so it is point-in-time on the event stream, while
    the article -> product_type_no mapping itself is a current-state snapshot
    (step 2.0). Ties broken deterministically so the set is reproducible.
    """
    prior = spark.table(ingest.TRANSACTIONS_TABLE).filter(
        F.col("t_dat") < F.lit(as_of).cast("date")
    )
    cats = spark.table(ingest.ARTICLES_TABLE).select("article_id", "product_type_no")
    prior_c = prior.join(F.broadcast(cats), "article_id", "inner")

    dominant = (
        prior_c.groupBy("customer_id", "product_type_no")
        .agg(F.count(F.lit(1)).alias("n"))
        .withColumn(
            "r",
            F.row_number().over(
                Window.partitionBy("customer_id").orderBy(
                    F.desc("n"), F.asc("product_type_no")
                )
            ),
        )
        .filter("r = 1")
        .select("customer_id", "product_type_no")
    )

    recent = prior_c.filter(
        F.col("t_dat") >= F.date_sub(F.lit(as_of).cast("date"), lookback_days)
    )
    pop = (
        recent.groupBy("product_type_no", "article_id")
        .agg(F.count(F.lit(1)).alias("n"))
        .withColumn(
            "source_rank",
            F.row_number().over(
                Window.partitionBy("product_type_no").orderBy(
                    F.desc("n"), F.asc("article_id")
                )
            ),
        )
        .filter(F.col("source_rank") <= max(n, 1))
        .select("product_type_no", "article_id", "source_rank")
    )

    return (
        dominant.join(F.broadcast(pop), "product_type_no", "inner")
        .select(
            "customer_id",
            "article_id",
            F.lit(SOURCE_CATEGORY).alias("source"),
            "source_rank",
        )
    )


def union_candidates(*sources: DataFrame) -> DataFrame:
    """
    Union the sources, keeping ONE row per (customer, article) but recording
    every source that produced it.

    Deduplicating to a single arbitrary source would destroy the diagnostic the
    tags exist for -- a candidate reachable from two sources is a different
    object from one reachable from a single source, and the ranker should be
    able to see that.
    """
    from functools import reduce

    all_rows = reduce(lambda a, b: a.unionByName(b), sources)
    return (
        all_rows.groupBy("customer_id", "article_id")
        .agg(
            F.collect_set("source").alias("sources"),
            F.min("source_rank").alias("best_source_rank"),
        )
        .withColumn("n_sources", F.size("sources"))
        .withColumn("from_ann", F.array_contains("sources", SOURCE_ANN))
        .withColumn("from_repurchase", F.array_contains("sources", SOURCE_REPURCHASE))
        .withColumn("from_category", F.array_contains("sources", SOURCE_CATEGORY))
    )


def recall_ceiling(candidates: DataFrame, truth: DataFrame) -> dict:
    """
    The hard ceiling on end-to-end recall, overall and per source.

    Denominator is true (customer, article) pairs -- the same denominator
    baselines.py and the two-tower evaluation use, so all three are comparable.
    """
    joined = truth.join(
        candidates, ["customer_id", "article_id"], "left"
    )
    agg = joined.agg(
        F.count(F.lit(1)).alias("n_true"),
        F.sum(F.when(F.col("n_sources").isNotNull(), 1).otherwise(0)).alias("covered"),
        F.sum(F.when(F.col("from_ann"), 1).otherwise(0)).alias("by_ann"),
        F.sum(F.when(F.col("from_repurchase"), 1).otherwise(0)).alias("by_repurchase"),
        F.sum(F.when(F.col("from_category"), 1).otherwise(0)).alias("by_category"),
    ).collect()[0]
    n = agg.n_true
    return {
        "n_true_pairs": n,
        "recall_ceiling": agg.covered / n,
        "by_ann": agg.by_ann / n,
        "by_repurchase": agg.by_repurchase / n,
        "by_category": agg.by_category / n,
    }


def candidate_set_stats(candidates: DataFrame) -> dict:
    per_cust = candidates.groupBy("customer_id").agg(F.count(F.lit(1)).alias("k"))
    row = per_cust.agg(
        F.count(F.lit(1)).alias("n_customers"),
        F.avg("k").alias("mean_k"),
        F.min("k").alias("min_k"),
        F.max("k").alias("max_k"),
        F.sum("k").alias("n_rows"),
    ).collect()[0]
    src = candidates.agg(
        F.sum(F.when(F.col("from_ann"), 1).otherwise(0)).alias("ann"),
        F.sum(F.when(F.col("from_repurchase"), 1).otherwise(0)).alias("repurchase"),
        F.sum(F.when(F.col("from_category"), 1).otherwise(0)).alias("category"),
        F.avg("n_sources").alias("mean_n_sources"),
    ).collect()[0]
    return {
        "n_customers": row.n_customers,
        "n_candidate_rows": row.n_rows,
        "mean_candidates_per_customer": float(row.mean_k),
        "min_candidates": row.min_k,
        "max_candidates": row.max_k,
        "rows_from_ann": src.ann,
        "rows_from_repurchase": src.repurchase,
        "rows_from_category": src.category,
        "mean_sources_per_candidate": float(src.mean_n_sources),
    }
