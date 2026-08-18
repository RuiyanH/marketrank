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
SOURCE_GLOBAL_POP = "global_pop"
SOURCE_COVISIT = "covisit"

N_REPURCHASE = 30
N_CATEGORY = 40
N_ANN = 50
N_GLOBAL_POP = 40
N_COVISIT = 40
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


def global_popularity_source(
    spark: SparkSession,
    as_of: str,
    n: int = N_GLOBAL_POP,
    lookback_days: int = B.POPULARITY_LOOKBACK_DAYS,
) -> DataFrame:
    """
    Plain global recent popularity, the same list `baselines.popularity_ranks`
    scores, cross-joined to every customer.

    Step 4.1 measured the doc's third source -- popularity inside the customer's
    DOMINANT category -- covering 2.162% of true pairs at depth 40, against
    3.168% for this at the same depth. The personalisation was COSTING recall: a
    single `product_type_no` is too narrow a cone. Adding this as a fourth
    source moved the ceiling 7.475% -> 9.083% for 27 more candidates per
    customer, and the reference build left it as an experiment. R.5 applies it.
    """
    pop = B.popularity_ranks(spark, as_of, lookback_days=lookback_days, depth=n)
    return pop.select(
        "article_id", F.col("pop_rank").alias("source_rank")
    ).withColumn("source", F.lit(SOURCE_GLOBAL_POP))


def cross_to_customers(source: DataFrame, customers: DataFrame) -> DataFrame:
    """Attach a customer-independent source (a global list) to every customer."""
    return customers.select("customer_id").distinct().crossJoin(F.broadcast(source))


def union_candidates(*sources: DataFrame, source_names: tuple[str, ...] = ()) -> DataFrame:
    """
    Union the sources, keeping ONE row per (customer, article) but recording
    every source that produced it.

    Deduplicating to a single arbitrary source would destroy the diagnostic the
    tags exist for -- a candidate reachable from two sources is a different
    object from one reachable from a single source, and the ranker should be
    able to see that.

    `source_names` drives the `from_<name>` boolean columns. It is a parameter
    rather than three hardcoded columns because R.5 adds a fourth and fifth
    source, and because the marginal-contribution table has to be able to build
    the union of an arbitrary SUBSET of the sources.
    """
    from functools import reduce

    names = source_names or (SOURCE_ANN, SOURCE_REPURCHASE, SOURCE_CATEGORY)
    all_rows = reduce(lambda a, b: a.unionByName(b), sources)
    out = (
        all_rows.groupBy("customer_id", "article_id")
        .agg(
            F.collect_set("source").alias("sources"),
            F.min("source_rank").alias("best_source_rank"),
        )
        .withColumn("n_sources", F.size("sources"))
    )
    for name in names:
        out = out.withColumn(f"from_{name}", F.array_contains("sources", name))
    return out


def recall_ceiling(
    candidates: DataFrame, truth: DataFrame, source_names: tuple[str, ...] = ()
) -> dict:
    """
    The hard ceiling on end-to-end recall, overall and per source.

    Denominator is true (customer, article) pairs -- the same denominator
    baselines.py and the two-tower evaluation use, so all three are comparable.
    """
    names = source_names or (SOURCE_ANN, SOURCE_REPURCHASE, SOURCE_CATEGORY)
    joined = truth.join(candidates, ["customer_id", "article_id"], "left")
    exprs = [
        F.count(F.lit(1)).alias("n_true"),
        F.sum(F.when(F.col("n_sources").isNotNull(), 1).otherwise(0)).alias("covered"),
    ]
    for name in names:
        exprs.append(
            F.sum(F.when(F.col(f"from_{name}"), 1).otherwise(0)).alias(f"by_{name}")
        )
    agg = joined.agg(*exprs).collect()[0]
    n = agg.n_true
    out = {"n_true_pairs": n, "recall_ceiling": agg.covered / n}
    for name in names:
        out[f"by_{name}"] = agg[f"by_{name}"] / n
    return out


def marginal_contribution(
    sources: dict[str, DataFrame], truth: DataFrame, budget_note: str = ""
) -> dict:
    """
    Each source's MARGINAL ceiling and its cost in candidate slots.

    Solo coverage is the wrong number once there is more than one source: what a
    source is worth is what the union LOSES when it is removed, and what it
    costs is the slots it occupies out of the fixed per-customer budget. The
    reference build measured mean sources per candidate at 1.033 -- the sources
    were nearly disjoint, so marginal was approximately solo -- but that stops
    being true as sources are added, which is exactly why this is the number to
    track rather than solo coverage.

    Returns, per source: solo ceiling, the full-union ceiling, the ceiling
    without it (`leave_one_out`), the marginal difference, the **marginal**
    slots per customer it costs, and **marginal ceiling per slot**, which is
    R.6's keep/demote criterion.

    Slots are counted MARGINALLY -- mean candidates per customer of the full
    union minus that of the union without this source -- not as the source's own
    row count. The two differ whenever sources overlap: a candidate produced by
    both co-visitation and repurchase occupies one slot, not two, so charging a
    source for its solo rows would overstate the cost of a source that mostly
    duplicates another. Marginal ceiling over marginal slots is the same
    accounting on both sides of the ratio, which is the only way the number
    means "what this source buys per slot it costs".

    Caveat worth stating rather than hiding: removing a source frees its slots,
    and this does NOT reallocate them to the survivors. So `marginal` is the
    value of the source at its current depth, not the value of the slots. A
    fully reallocating version would have to re-tune every other source's depth,
    which makes it a search rather than a measurement.
    """
    def mean_k(df: DataFrame) -> float:
        return float(
            df.groupBy("customer_id").agg(F.count(F.lit(1)).alias("k"))
            .agg(F.avg("k").alias("m")).collect()[0].m
        )

    def reach(df: DataFrame) -> int:
        return df.select("customer_id").distinct().count()

    names = tuple(sources)
    all_union = union_candidates(*sources.values(), source_names=names)
    full = recall_ceiling(all_union, truth, source_names=names)
    per_customer = mean_k(all_union)
    cohort_n = reach(all_union)

    out = {
        "budget_note": budget_note,
        "union_ceiling": full["recall_ceiling"],
        "n_true_pairs": full["n_true_pairs"],
        "mean_candidates_per_customer": float(per_customer),
        "sources": {},
    }
    for name in names:
        solo = recall_ceiling(
            union_candidates(sources[name], source_names=(name,)),
            truth, source_names=(name,),
        )
        rest = {k: v for k, v in sources.items() if k != name}
        if rest:
            loo_union = union_candidates(*rest.values(), source_names=tuple(rest))
            loo = recall_ceiling(loo_union, truth, source_names=tuple(rest))[
                "recall_ceiling"
            ]
            loo_k = mean_k(loo_union)
        else:
            loo, loo_k = 0.0, 0.0
        marginal = full["recall_ceiling"] - loo
        slots = per_customer - loo_k
        # REACH is not decoration. `solo` is a ceiling over the WHOLE cohort,
        # while the slot average below covers only the customers a source
        # actually reaches -- different denominators, so dividing one by the
        # other compares sources unfairly. Co-visitation is the case that
        # exposes it: ~46% reach at a 30-day lookback, so its slot efficiency
        # partly reflects spending slots only where it has signal. R.6's rule
        # reads `marginal_per_slot`, whose terms are both cohort-wide, which is
        # why the decision is unaffected -- but the trap belongs named, not
        # sitting unlabelled in the artifact.
        n_reached = reach(sources[name])
        out["sources"][name] = {
            "solo": solo["recall_ceiling"],
            "reach_customers": n_reached,
            "reach_frac": (n_reached / cohort_n) if cohort_n else float("nan"),
            "solo_slots_per_covered_customer": mean_k(sources[name]),
            "leave_one_out": loo,
            "marginal": marginal,
            "marginal_slots": slots,
            "marginal_per_slot": (marginal / slots) if slots > 0 else float("nan"),
        }
    return out


def candidate_set_stats(
    candidates: DataFrame, source_names: tuple[str, ...] = ()
) -> dict:
    names = source_names or (SOURCE_ANN, SOURCE_REPURCHASE, SOURCE_CATEGORY)
    per_cust = candidates.groupBy("customer_id").agg(F.count(F.lit(1)).alias("k"))
    row = per_cust.agg(
        F.count(F.lit(1)).alias("n_customers"),
        F.avg("k").alias("mean_k"),
        F.min("k").alias("min_k"),
        F.max("k").alias("max_k"),
        F.sum("k").alias("n_rows"),
    ).collect()[0]
    exprs = [F.avg("n_sources").alias("mean_n_sources")]
    for name in names:
        exprs.append(
            F.sum(F.when(F.col(f"from_{name}"), 1).otherwise(0)).alias(f"rows_{name}")
        )
    src = candidates.agg(*exprs).collect()[0]
    out = {
        "n_customers": row.n_customers,
        "n_candidate_rows": row.n_rows,
        "mean_candidates_per_customer": float(row.mean_k),
        "min_candidates": row.min_k,
        "max_candidates": row.max_k,
        "mean_sources_per_candidate": float(src.mean_n_sources),
    }
    for name in names:
        out[f"rows_from_{name}"] = src[f"rows_{name}"]
    return out
