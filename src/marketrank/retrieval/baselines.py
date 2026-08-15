"""
Retrieval baselines, measured before the two-tower model exists.

On H&M the naive baselines are *strong*. Repeat purchase is a large fraction of
the signal and recent popularity is roughly what the competition's median entry
amounted to, so a two-tower model's job is to beat **the union of both**, not to
beat random. Finding that out in week 3 is cheap; finding it out in week 6 is
not, and "recall@N up x% over baseline" is a lie if the baseline was chosen to be
beatable.

Recall is measured WITHOUT materialising candidate lists. For each true
(customer, article) purchase pair in the evaluation window, the pair's rank in
each baseline's list is computable directly:

* repurchase rank  = the article's recency rank among that customer's distinct
                     prior purchases (null if never bought)
* popularity rank  = the article's position in the global recent-popularity list
                     (null if outside the top P)
* union rank       = repurchase rank if present, else
                     (that customer's repurchase-list length) + popularity rank

So `recall@N` is `mean(rank <= N)` over the true pairs. Exact, and it never
builds a 300k x 500 table.

Everything is computed from `t_dat <= as_of - 1 day`; the evaluation window
starts at `as_of`. Note this fixes the candidate list at the start of a 14-day
window rather than recomputing it daily -- a standard retrieval-eval
simplification, and PIT-safe in the direction that matters: it never uses the
future.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window, functions as F

from marketrank import ingest, splits

POPULARITY_LOOKBACK_DAYS = 30
POPULARITY_DEPTH = 2000


def truth_pairs(spark: SparkSession, slice_name: str = "val_tune") -> DataFrame:
    """Distinct (customer_id, article_id) purchased inside a split slice."""
    lo, hi = splits.bounds(slice_name)
    return (
        spark.table(ingest.TRANSACTIONS_TABLE)
        .filter(F.col("t_dat").between(F.lit(lo).cast("date"), F.lit(hi).cast("date")))
        .select("customer_id", "article_id")
        .distinct()
    )


def popularity_ranks(
    spark: SparkSession,
    as_of: str,
    lookback_days: int = POPULARITY_LOOKBACK_DAYS,
    depth: int = POPULARITY_DEPTH,
) -> DataFrame:
    """(article_id, pop_rank) for the top `depth` articles in the lookback window."""
    txn = spark.table(ingest.TRANSACTIONS_TABLE).filter(
        F.col("t_dat") < F.lit(as_of).cast("date")
    )
    txn = txn.filter(
        F.col("t_dat") >= F.date_sub(F.lit(as_of).cast("date"), lookback_days)
    )
    counts = txn.groupBy("article_id").agg(F.count(F.lit(1)).alias("n"))
    w = Window.orderBy(F.desc("n"), F.asc("article_id"))
    return (
        counts.withColumn("pop_rank", F.row_number().over(w))
        .filter(F.col("pop_rank") <= depth)
        .select("article_id", "pop_rank")
    )


def repurchase_ranks(spark: SparkSession, as_of: str) -> DataFrame:
    """
    (customer_id, article_id, rep_rank) -- the customer's own prior purchases,
    most-recent-first, deduplicated to one row per article.
    """
    txn = spark.table(ingest.TRANSACTIONS_TABLE).filter(
        F.col("t_dat") < F.lit(as_of).cast("date")
    )
    last_seen = txn.groupBy("customer_id", "article_id").agg(
        F.max("t_dat").alias("last_dat"), F.count(F.lit(1)).alias("n_bought")
    )
    w = Window.partitionBy("customer_id").orderBy(
        F.desc("last_dat"), F.desc("n_bought"), F.asc("article_id")
    )
    return last_seen.withColumn("rep_rank", F.row_number().over(w)).select(
        "customer_id", "article_id", "rep_rank"
    )


def rank_truth(
    spark: SparkSession, slice_name: str = "val_tune", **kwargs
) -> DataFrame:
    """
    Attach each baseline's rank to every true purchase pair.

    Returns (customer_id, article_id, rep_rank, pop_rank, union_rank), with nulls
    where a baseline does not retrieve the article at all.
    """
    as_of = splits.bounds(slice_name)[0]
    truth = truth_pairs(spark, slice_name)
    rep = repurchase_ranks(spark, as_of, **{})
    pop = popularity_ranks(spark, as_of, **kwargs)

    rep_len = rep.groupBy("customer_id").agg(F.max("rep_rank").alias("rep_len"))

    out = (
        truth.join(rep, ["customer_id", "article_id"], "left")
        .join(F.broadcast(pop), ["article_id"], "left")
        .join(rep_len, ["customer_id"], "left")
    )
    out = out.withColumn("rep_len", F.coalesce(F.col("rep_len"), F.lit(0)))
    return out.withColumn(
        "union_rank",
        F.when(F.col("rep_rank").isNotNull(), F.col("rep_rank")).otherwise(
            F.col("rep_len") + F.col("pop_rank")
        ),
    ).select("customer_id", "article_id", "rep_rank", "pop_rank", "rep_len", "union_rank")


def recall_at(ranked: DataFrame, ns: tuple[int, ...] = (100, 500)) -> dict:
    """recall@N per baseline, plus the denominator."""
    total = ranked.count()
    exprs = [F.count(F.lit(1)).alias("n_pairs")]
    for col in ("rep_rank", "pop_rank", "union_rank"):
        for n in ns:
            exprs.append(
                F.sum(
                    F.when(F.col(col).isNotNull() & (F.col(col) <= n), 1).otherwise(0)
                ).alias(f"{col}_at_{n}")
            )
    row = ranked.agg(*exprs).collect()[0].asDict()
    out = {"n_true_pairs": total}
    for k, v in row.items():
        if k != "n_pairs":
            out[k] = v / total if total else float("nan")
    return out
