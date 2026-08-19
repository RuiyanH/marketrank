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

# Step R.2 -- the same rolling aggregates, article side, fed to the ARTICLE
# tower. These already existed from week 2 and were simply never wired in.
#
# This is the fix with the highest prior in the recovery plan, and the reason is
# an information asymmetry rather than a modelling idea: the popularity baseline
# that beat the tower sees recent transaction volume, and the article tower was
# an id plus five STATIC categorical embeddings -- structurally unable to
# represent "trending now" on a fast-fashion dataset where the measured
# short-horizon signal is trend rather than identity (exact-article repurchase
# ceiling 3.36% against 64.34% at product_type_no; step 3.1).
#
# THE COST, STATED UP FRONT: this makes the article vector time-varying. The
# article index stops being exportable once and has to be re-exported as of the
# scoring date, and the week-6 serving path inherits exactly that cost.
ARTICLE_NUMERIC = [
    f"art_{m}_{w}d"
    for w in ft.WINDOWS
    for m in ("n_txn", "n_customers", "spend", "avg_price")
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


def with_article_volume(df: DataFrame, spark: SparkSession, on_day: str = "day_index"):
    """
    Attach the article's rolling volume features, log1p'd, joined PIT.

    `feature_article_daily`'s row for (article, d) is already stamped "as of end
    of day d-1" by the week-2 window frame, so joining it on the event's own
    `day_index` is a plain equi-join and carries the same PIT guarantee the
    customer side has. No new rule, new place.

    Missing -> 0 after log1p, which is the honest value: an article with no row
    for that day sold nothing in the window. That is only true because R.1's
    rebuild put an article spine on the scoring day; without it the zeros would
    be indistinguishable from "no data" (BUILD_NOTES step R.0).
    """
    feats = spark.table(ft.ARTICLE_FEATURE_TABLE).select(
        "article_id", F.col("day_index").alias("_art_day"), *ARTICLE_NUMERIC
    )
    out = df.join(
        feats,
        (df["article_id"] == feats["article_id"]) & (df[on_day] == feats["_art_day"]),
        "left",
    ).drop(feats["article_id"]).drop("_art_day")
    for c in ARTICLE_NUMERIC:
        out = out.withColumn(
            c, F.log1p(F.coalesce(F.col(c).cast("double"), F.lit(0.0)))
        )
    return out


def article_frame(
    spark: SparkSession, vocabs: dict, as_of_day: int | None = None
) -> DataFrame:
    """
    One row per article in the catalog: (article_id, article_idx, categoricals),
    plus the rolling volume features **as of `as_of_day`** when given.

    `as_of_day` is a `day_index`. Passing None reproduces the pre-R.2 static
    frame with the volume columns zero-filled, which is what the R.1 ablation
    rung needs to stay comparable.
    """
    arts = spark.table(ingest.ARTICLES_TABLE)
    out = with_article_idx(arts, article_index_df(spark, vocabs))
    for c in ARTICLE_CATEGORICALS:
        out = out.withColumn(f"{c}_idx", _map_col(c, vocabs[c]))
    if as_of_day is None:
        for c in ARTICLE_NUMERIC:
            out = out.withColumn(c, F.lit(0.0))
    else:
        out = out.withColumn("_as_of", F.lit(as_of_day))
        out = with_article_volume(out, spark, on_day="_as_of").drop("_as_of")
    return out.select(
        "article_id",
        "article_idx",
        *[f"{c}_idx" for c in ARTICLE_CATEGORICALS],
        *ARTICLE_NUMERIC,
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
    train_start: str | None = "2020-02-14",
    train_end: str | None = None,
    eval_customer_sample: int = 20_000,
    article_volume: bool = False,
) -> dict:
    """
    Write vocabs, the article frame, the training rows and the eval rows.

    `article_volume` (step R.2) attaches the article's rolling volume features
    to both the training rows and the article index, the latter **as of the
    scoring day**. Left False, the volume columns are present but zero, so the
    parquet schema is stable across the ablation ladder and the R.1 rung stays
    bit-comparable with the reference build.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Symmetric with train_end below. "auto"/None means the train SLICE's own
    # bound, so the full-scale window is never a date typed into a job script
    # that can drift away from splits.py. The literal default stays
    # 2020-02-14 -- the laptop ladder's reduced window -- so re-exporting an
    # earlier rung reproduces it rather than silently widening it.
    if train_start in (None, "auto"):
        train_start = splits.bounds("train")[0]
    train_end = train_end or splits.bounds("train")[1]

    vocabs = build_vocabs(spark)
    (out_dir / "vocabs.json").write_text(
        json.dumps({k: {str(a): b for a, b in v.items()} for k, v in vocabs.items()})
    )

    # Evaluation cohort: customers who bought during val_tune, scored as of the
    # first day of val_tune. day0 is needed BEFORE the article frame is written,
    # because from R.2 on the article index is exported as of the scoring day
    # rather than once ever.
    lo, hi = splits.bounds("val_tune")
    day0 = (
        spark.sql(f"SELECT datediff(date'{lo}', date'{ft.DAY_ZERO}') AS d")
        .collect()[0]
        .d
    )

    arts = article_frame(spark, vocabs, as_of_day=day0 if article_volume else None)
    arts.coalesce(1).write.mode("overwrite").parquet(str(out_dir / "articles"))

    events = training_events(spark, customer_sample, train_start, train_end)
    aidx = article_index_df(spark, vocabs)
    events = with_article_idx(events, aidx)
    if article_volume:
        events = with_article_volume(events, spark)
    else:
        for c in ARTICLE_NUMERIC:
            events = events.withColumn(c, F.lit(0.0))
    train = customer_context(spark, events, vocabs).drop("customer_id", "article_id")
    train.write.mode("overwrite").parquet(str(out_dir / "train"))
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

    # python -m marketrank.retrieval.dataset [COHORT] [--article-volume] [--out DIR]
    #
    # `--out` exists so an R.2 export does not overwrite the R.1 one. Every rung
    # of the recovery ladder has to stay re-verifiable from disk, and an export
    # written in place takes its own rung's numbers with it.
    _n = int(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else 100_000
    # COHORT <= 0 means "no cap", i.e. every customer. `training_events` guards on
    # `is not None`, so passing 0 straight through would `.limit(0)` and export an
    # EMPTY training set -- a full-scale run that silently trains on nothing. The
    # CLI previously had no way to express None at all.
    _n = None if _n <= 0 else _n
    _vol = "--article-volume" in sys.argv
    _out = DATASET_DIR
    if "--out" in sys.argv:
        _out = Path(sys.argv[sys.argv.index("--out") + 1])
    # TRAINING WINDOW. Hardcoded at the laptop's 2020-02-14 and unreachable from
    # the CLI, which made "full scale" undeliverable: `dataset 0` lifted the
    # COHORT cap only, so R.4's export ran 1.37M customers over six months and
    # produced 7,117,442 rows where the rung needs ~28M. Both caps have to lift.
    # `--train-start auto` takes the train slice's own lower bound.
    _start = "2020-02-14"
    if "--train-start" in sys.argv:
        _start = sys.argv[sys.argv.index("--train-start") + 1]
    _spark = get_spark("twotower_export", driver_memory="10g")
    _t = time.time()
    _stats = export(_spark, out_dir=_out, customer_sample=_n,
                    train_start=_start, article_volume=_vol)
    print("EXPORT out_dir", _out)
    print("EXPORT article_volume", _vol)
    print("EXPORT train_start", _start)
    print("EXPORT_SECONDS %.1f" % (time.time() - _t))
    for k, v in _stats.items():
        print("EXPORT", k, v)
    _spark.stop()
