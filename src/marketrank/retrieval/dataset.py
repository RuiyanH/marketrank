"""
Build the two-tower training set in Spark and land it as parquet.

Everything here is PIT. The customer side of a training example for an event on
day `d` is built from:

* `feature_customer_daily` at `(customer, d)` -- already stamped "as of d-1" by
  the window frame (week 2), so this is a plain equi-join;
* the customer's recent articles over `[d-90, d-1]`, from a range frame with the
  same `-1` upper bound.

"Recent articles" means recent **as of d-1**. Same rule as week 2, new place, and
this is exactly where a leak sneaks back in.

The customer's *static* attributes (age, club status, newsletter frequency) are
NOT point-in-time -- `customers.csv` is a current-state snapshot with no history.
Stated in the README's limitations; unavoidable with this dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window, functions as F

from marketrank import config, features as ft, ingest, splits

DATASET_DIR = config.PROJECT_ROOT / "artifacts" / "twotower"

# How many recent articles feed the customer tower's item-average.
RECENT_K = 20
# The history window the recent-article list is drawn from -- the same 90 days
# the rolling features use, so one PIT rule covers both.
RECENT_DAYS = ft.MAX_WINDOW

ARTICLE_CATEGORICALS = [
    "product_type_no",
    "colour_group_code",
    "department_no",
    "index_group_no",
    "garment_group_no",
]
CUSTOMER_CATEGORICALS = ["club_member_status", "fashion_news_frequency"]

# Rolling features fed to the customer tower, log1p'd because they are counts and
# spends with very long tails.
CUSTOMER_NUMERIC = [
    f"cust_{m}_{w}d"
    for w in ft.WINDOWS
    for m in ("n_txn", "n_articles", "spend", "avg_price")
]

AGE_BUCKETS = [20, 25, 30, 35, 40, 50, 60, 70]


def _vocab(values: list) -> dict:
    """Value -> index, with 0 reserved for unknown/null."""
    return {v: i + 1 for i, v in enumerate(sorted(v for v in values if v is not None))}


def build_vocabs(spark: SparkSession) -> dict:
    arts = spark.table(ingest.ARTICLES_TABLE)
    custs = spark.table(ingest.CUSTOMERS_TABLE)
    vocabs = {
        "article_id": _vocab(
            [r.article_id for r in arts.select("article_id").distinct().collect()]
        )
    }
    for c in ARTICLE_CATEGORICALS:
        vocabs[c] = _vocab([r[0] for r in arts.select(c).distinct().collect()])
    for c in CUSTOMER_CATEGORICALS:
        vocabs[c] = _vocab([r[0] for r in custs.select(c).distinct().collect()])
    return vocabs


def _map_col(col: str, vocab: dict):
    """
    Spark expression mapping a column through a python dict, null/unseen -> 0.

    Only for SMALL vocabularies (the categoricals, all under 500 values). The
    105,542-entry article vocabulary goes through `article_index_df` and a join
    instead: a `create_map` literal of that size is 211k expressions in the query
    plan and Spark's codegen chokes on it.
    """
    return F.coalesce(
        F.create_map([F.lit(x) for kv in vocab.items() for x in kv])[F.col(col)],
        F.lit(0),
    )


def article_index_df(spark: SparkSession, vocabs: dict) -> DataFrame:
    """(article_id, article_idx) as a broadcastable DataFrame."""
    rows = [(k, v) for k, v in vocabs["article_id"].items()]
    return spark.createDataFrame(rows, "article_id string, article_idx int")


def with_article_idx(df: DataFrame, aidx: DataFrame) -> DataFrame:
    return df.join(F.broadcast(aidx), "article_id", "left").withColumn(
        "article_idx", F.coalesce(F.col("article_idx"), F.lit(0))
    )


def age_bucket(col: str = "age"):
    e = F.when(F.col(col).isNull(), F.lit(0))
    for i, b in enumerate(AGE_BUCKETS):
        e = e.when(F.col(col) < b, F.lit(i + 1))
    return e.otherwise(F.lit(len(AGE_BUCKETS) + 1))


def article_frame(spark: SparkSession, vocabs: dict) -> DataFrame:
    """(article_idx, <categorical idx>...) -- one row per article in the catalog."""
    arts = spark.table(ingest.ARTICLES_TABLE)
    out = with_article_idx(arts, article_index_df(spark, vocabs))
    for c in ARTICLE_CATEGORICALS:
        out = out.withColumn(f"{c}_idx", _map_col(c, vocabs[c]))
    return out.select(
        "article_id", "article_idx", *[f"{c}_idx" for c in ARTICLE_CATEGORICALS]
    )


def customer_context(
    spark: SparkSession,
    events: DataFrame,
    vocabs: dict,
) -> DataFrame:
    """
    Attach the PIT customer context to `events` (customer_id, day_index, ...).

    `events` must already carry `day_index`. The recent-article list is built
    from the full transaction log with a `rangeBetween(-RECENT_DAYS, -1)` frame,
    so it never sees the event's own day.
    """
    txn = ft.with_day_index(spark.table(ingest.TRANSACTIONS_TABLE))
    hist = with_article_idx(txn, article_index_df(spark, vocabs)).select(
        "customer_id", "day_index", "article_idx"
    )

    # One row per (customer, day) that we need context for, unioned with the
    # customer's own transaction days so the frame has something to look at.
    need = events.select("customer_id", "day_index").distinct()
    frame_rows = hist.unionByName(
        need.withColumn("article_idx", F.lit(None).cast("int"))
    )

    w = (
        Window.partitionBy("customer_id")
        .orderBy("day_index")
        .rangeBetween(-RECENT_DAYS, -1)  # -1. Same rule as week 2.
    )
    recent = (
        frame_rows.withColumn("hist", F.collect_list("article_idx").over(w))
        .select("customer_id", "day_index", "hist")
        .distinct()
    )
    # Most-recent-last in the frame, so take the tail and reverse.
    recent = recent.withColumn(
        "recent_articles",
        F.reverse(F.slice(F.col("hist"), -RECENT_K, RECENT_K)),
    ).drop("hist")

    custs = spark.table(ingest.CUSTOMERS_TABLE)
    custs = custs.withColumn("age_bucket", age_bucket())
    for c in CUSTOMER_CATEGORICALS:
        custs = custs.withColumn(f"{c}_idx", _map_col(c, vocabs[c]))
    custs = custs.select(
        "customer_id", "age_bucket", *[f"{c}_idx" for c in CUSTOMER_CATEGORICALS]
    )

    feats = spark.table(ft.CUSTOMER_FEATURE_TABLE).select(
        "customer_id", "day_index", *CUSTOMER_NUMERIC
    )

    out = (
        events.join(recent, ["customer_id", "day_index"], "left")
        .join(F.broadcast(custs), "customer_id", "left")
        .join(feats, ["customer_id", "day_index"], "left")
    )
    for c in CUSTOMER_NUMERIC:
        out = out.withColumn(c, F.log1p(F.coalesce(F.col(c).cast("double"), F.lit(0.0))))
    out = out.withColumn(
        "recent_articles",
        F.coalesce(F.col("recent_articles"), F.array().cast("array<int>")),
    )
    return out.fillna(0, subset=["age_bucket"] + [f"{c}_idx" for c in CUSTOMER_CATEGORICALS])


def training_events(
    spark: SparkSession,
    customer_sample: int | None,
    start: str,
    end: str,
    seed: int = 17,
) -> DataFrame:
    """
    Positives: one row per (customer, article, day) purchase in [start, end].

    `customer_sample` caps the cohort -- see BUILD_NOTES for why this build runs
    a reduced cohort and what a full run would need.
    """
    txn = ft.with_day_index(spark.table(ingest.TRANSACTIONS_TABLE)).filter(
        F.col("t_dat").between(F.lit(start).cast("date"), F.lit(end).cast("date"))
    )
    if customer_sample is not None:
        cohort = (
            txn.select("customer_id")
            .distinct()
            .withColumn("h", F.abs(F.hash(F.col("customer_id"), F.lit(seed))))
            .orderBy("h")
            .limit(customer_sample)
            .select("customer_id")
        )
        txn = txn.join(F.broadcast(cohort), "customer_id", "inner")
    return txn.select("customer_id", "article_id", "day_index", "feature_date").distinct()


def export(
    spark: SparkSession,
    out_dir: Path = DATASET_DIR,
    customer_sample: int | None = 100_000,
    train_start: str = "2020-02-14",
    train_end: str | None = None,
    eval_customer_sample: int = 20_000,
) -> dict:
    """Write vocabs, the article frame, the training rows and the eval rows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    train_end = train_end or splits.bounds("train")[1]

    vocabs = build_vocabs(spark)
    (out_dir / "vocabs.json").write_text(
        json.dumps({k: {str(a): b for a, b in v.items()} for k, v in vocabs.items()})
    )

    arts = article_frame(spark, vocabs)
    arts.coalesce(1).write.mode("overwrite").parquet(str(out_dir / "articles"))

    events = training_events(spark, customer_sample, train_start, train_end)
    aidx = article_index_df(spark, vocabs)
    events = with_article_idx(events, aidx)
    train = customer_context(spark, events, vocabs).drop("customer_id", "article_id")
    train.write.mode("overwrite").parquet(str(out_dir / "train"))

    # Evaluation cohort: customers who bought during val_tune, scored as of the
    # first day of val_tune.
    lo, hi = splits.bounds("val_tune")
    day0 = (
        spark.sql(f"SELECT datediff(date'{lo}', date'{ft.DAY_ZERO}') AS d")
        .collect()[0]
        .d
    )
    truth = (
        spark.table(ingest.TRANSACTIONS_TABLE)
        .filter(F.col("t_dat").between(F.lit(lo).cast("date"), F.lit(hi).cast("date")))
        .select("customer_id", "article_id")
        .distinct()
    )
    eval_cohort = (
        truth.select("customer_id")
        .distinct()
        .withColumn("h", F.abs(F.hash(F.col("customer_id"), F.lit(99))))
        .orderBy("h")
        .limit(eval_customer_sample)
        .select("customer_id")
    )
    eval_events = eval_cohort.withColumn("day_index", F.lit(day0))
    eval_ctx = customer_context(spark, eval_events, vocabs)
    eval_ctx.coalesce(4).write.mode("overwrite").parquet(str(out_dir / "eval_customers"))

    eval_truth = with_article_idx(
        truth.join(F.broadcast(eval_cohort), "customer_id", "inner"), aidx
    )
    eval_truth.coalesce(4).write.mode("overwrite").parquet(str(out_dir / "eval_truth"))

    return {
        "n_articles": arts.count(),
        "n_train_rows": spark.read.parquet(str(out_dir / "train")).count(),
        "n_eval_customers": eval_customer_sample,
        "n_eval_truth_pairs": eval_truth.count(),
    }


if __name__ == "__main__":
    import sys
    import time

    from marketrank.spark import get_spark

    _n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    _spark = get_spark("twotower_export", driver_memory="10g")
    _t = time.time()
    _stats = export(_spark, customer_sample=_n)
    print("EXPORT_SECONDS %.1f" % (time.time() - _t))
    for k, v in _stats.items():
        print("EXPORT", k, v)
    _spark.stop()
