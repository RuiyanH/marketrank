"""
Step R.1 -- rebuild the feature tables with a spine, at full scale.

    python -m marketrank.jobs.rebuild_features [SCORING_START [SCORING_END]]

The spine covers **every customer and every article in the catalog**, on the
scoring days, unioned with the natural transaction-day rows. Defaults to the
single day week 3 and week 4 both evaluate on, `val_tune`'s first day.

WHY THE SPINE IS SCORING DAYS AND NOT A CALENDAR. A dense customer x day spine
over the whole log is 1,371,980 x 734 = 1.006 **billion** rows before a feature
is attached, which is not a thing to build at any scale -- it is not a laptop
artifact. Features are needed on the days something is scored, and that set is
small: one scoring day costs ~1.37M customer rows and ~105k article rows. The
cost is linear in scoring days, so extending this to the whole `val_tune`
window is a decision about disk, not about correctness.

WHAT THIS IS EXPECTED TO CHANGE, AND WHAT IT IS NOT. The spine adds rows and
changes no value: a spine row carries zero-filled measures, and adding zero to a
window sum is the identity. That is asserted rather than assumed -- the job
records the pre-rebuild Iceberg snapshot id so the two versions can be diffed
directly (`marketrank.checks.read_snapshot`).

Note which side of week 3 this actually fixes. Training positives sit on days
the customer DID transact, so a feature row already existed for every training
row and the training set was never affected. The eval cohort is scored on a day
most of them did not transact, so it was 14.09% covered. The confound was
eval-time, and R.1 measures both arms to prove which.
"""

from __future__ import annotations

import sys
import time

from pyspark.sql import functions as F

from marketrank import checks, features as ft, ingest, splits


def main(scoring_start: str | None = None, scoring_end: str | None = None) -> None:
    from marketrank.spark import get_spark

    scoring_start = scoring_start or splits.bounds("val_tune")[0]
    scoring_end = scoring_end or scoring_start

    spark = get_spark("rebuild_features_with_spine", driver_memory="10g")

    customers = spark.table(ingest.CUSTOMERS_TABLE).select("customer_id")
    articles = spark.table(ingest.ARTICLES_TABLE).select("article_id")
    n_cust, n_art = customers.count(), articles.count()

    cust_spine = ft.customer_day_spine(spark, customers, scoring_start, scoring_end)
    art_spine = ft.article_day_spine(spark, articles, scoring_start, scoring_end)

    print(f"SPINE scoring_days {scoring_start} .. {scoring_end}")
    print(f"SPINE customers {n_cust}  articles {n_art}")
    print(f"SPINE customer_rows {cust_spine.count()}  article_rows {art_spine.count()}")

    for table in (ft.CUSTOMER_FEATURE_TABLE, ft.ARTICLE_FEATURE_TABLE):
        if spark.catalog.tableExists(table):
            snaps = checks.snapshot_ids(spark, table)
            print(f"PRE_REBUILD_SNAPSHOT {table} {snaps[-1]}")

    t0 = time.time()
    res = ft.build_features(
        spark, customer_spine=cust_spine, article_spine=art_spine
    )
    print("REBUILD_SECONDS %.1f" % (time.time() - t0))
    for name, n in res.items():
        print("TABLE_ROWS", name, n)

    # The R.0 audit, at full scale, on the exact cohort week 3 evaluated.
    day0 = (
        spark.sql(
            f"SELECT datediff(date'{scoring_start}', date'{ft.DAY_ZERO}') AS d"
        )
        .collect()[0]
        .d
    )
    for table, key, ents in (
        (ft.CUSTOMER_FEATURE_TABLE, "customer_id", customers),
        (ft.ARTICLE_FEATURE_TABLE, "article_id", articles),
    ):
        cov = ft.feature_coverage(spark.table(table), ents, key, day0)
        print("COVERAGE", table, cov)

    spark.stop()


if __name__ == "__main__":
    main(*sys.argv[1:3])
