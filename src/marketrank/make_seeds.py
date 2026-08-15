"""
Generate dbt's committed seed CSVs from the real warehouse.

CI has no 3.5 GB CSV and no warehouse, so the dimensional model and its tests
run on a few hundred committed rows instead. Those rows are sampled here rather
than typed by hand so that they are real data with real edge cases, and the
selection is deterministic so re-running it produces the same seeds.

Run once, commit the output:

    python -m marketrank.make_seeds

The seeds must satisfy the tests they are meant to exercise, so the selection is
closed under referential integrity (every fact article/customer is in the
dimensions) and deliberately includes a multi-quantity purchase and a null age.
"""

import csv
from pathlib import Path

from pyspark.sql import functions as F

from marketrank import config, ingest
from marketrank.spark import get_spark

SEED_DIR = config.PROJECT_ROOT / "dbt" / "seeds"

N_CUSTOMERS = 50
N_ARTICLES = 50
N_TRANSACTIONS = 400  # cap; the real count falls out of the customer choice

# A fixed window keeps the sample small and its provenance obvious.
WINDOW_START = "2020-09-01"
WINDOW_END = "2020-09-22"

ARTICLE_COLS = [
    "article_id", "product_code", "prod_name", "product_type_no",
    "product_type_name", "product_group_name", "graphical_appearance_no",
    "graphical_appearance_name", "colour_group_code", "colour_group_name",
    "perceived_colour_value_id", "perceived_colour_value_name",
    "perceived_colour_master_id", "perceived_colour_master_name",
    "department_no", "department_name", "index_code", "index_name",
    "index_group_no", "index_group_name", "section_no", "section_name",
    "garment_group_no", "garment_group_name", "detail_desc",
]
CUSTOMER_COLS = [
    "customer_id", "FN", "Active", "club_member_status",
    "fashion_news_frequency", "age", "postal_code",
]
TRANSACTION_COLS = [
    "t_dat", "customer_id", "article_id", "price", "sales_channel_id",
    "_ingested_at",
]


def _write(path: Path, header: list[str], rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow(["" if v is None else v for v in r])
    print(f"wrote {path} ({len(rows)} rows)")


def main() -> None:
    spark = get_spark("make_seeds")
    txn = (
        spark.table(ingest.TRANSACTIONS_TABLE)
        .filter(F.col("t_dat").between(WINDOW_START, WINDOW_END))
    )

    # 1. Articles first -- the most-purchased ones in the window. Choosing the
    #    dimension before the fact is what keeps the sample referentially closed
    #    without throwing most of the transactions away.
    art_ids = [
        r.article_id
        for r in (
            txn.groupBy("article_id").count()
            .orderBy(F.desc("count"), F.asc("article_id"))
            .limit(N_ARTICLES)
            .collect()
        )
    ]
    on_articles = txn.filter(F.col("article_id").isin(art_ids))

    # 2. Customers with a genuine multi-quantity purchase of one of those
    #    articles. These are the rows the grain decision exists for.
    multi_ids = [
        r.customer_id
        for r in (
            on_articles
            .groupBy("customer_id", "article_id", "t_dat", "sales_channel_id")
            .count()
            .filter(F.col("count") > 1)
            .select("customer_id")
            .distinct()
            .orderBy("customer_id")
            .limit(8)
            .collect()
        )
    ]

    # 3. Top up to N_CUSTOMERS with other customers who bought those articles.
    #    Moderately active customers (3-8 purchases in the window), not the
    #    heaviest: the seed needs several rows per customer for the tests to
    #    have depth, but taking the head of the distribution would make the
    #    multi-quantity rate wildly unrepresentative of the real 10%.
    other_ids = [
        r.customer_id
        for r in (
            on_articles.groupBy("customer_id").count()
            .filter(~F.col("customer_id").isin(multi_ids))
            .filter(F.col("count").between(3, 8))
            .orderBy(F.desc("count"), F.asc("customer_id"))
            .limit(N_CUSTOMERS * 2)
            .collect()
        )
    ]
    cust_ids = (multi_ids + [c for c in other_ids if c not in multi_ids])[
        : N_CUSTOMERS - 1  # one slot reserved for a null age
    ]

    sel = on_articles.filter(F.col("customer_id").isin(cust_ids))

    rows = (
        sel.orderBy("t_dat", "customer_id", "article_id", "sales_channel_id", "price")
        .limit(N_TRANSACTIONS)
        .collect()
    )
    txn_rows = [
        (r.t_dat.isoformat(), r.customer_id, r.article_id, r.price,
         r.sales_channel_id, "2026-08-14 00:00:00")
        for r in rows
    ]

    n_multi = len(rows) - len({
        (r.customer_id, r.article_id, r.t_dat, r.sales_channel_id) for r in rows
    })
    print(f"multi-quantity duplicate rows surviving the cut: {n_multi}")
    if n_multi == 0:
        raise SystemExit("seeds must contain a multi-quantity purchase")

    kept_customers = sorted({r.customer_id for r in rows})
    kept_articles = sorted({r.article_id for r in rows})

    # 4. One customer with a null age, guaranteed. They have no transactions,
    #    which is also realistic -- most customers do not buy in a given month.
    null_age = (
        spark.table(ingest.CUSTOMERS_TABLE)
        .filter(F.col("age").isNull())
        .orderBy("customer_id")
        .limit(1)
        .collect()[0]
        .customer_id
    )
    customer_ids = sorted(set(kept_customers) | {null_age})

    cust_rows = [
        tuple(r[c] for c in CUSTOMER_COLS)
        for r in (
            spark.table(ingest.CUSTOMERS_TABLE)
            .filter(F.col("customer_id").isin(customer_ids))
            .orderBy("customer_id")
            .collect()
        )
    ]
    n_null_age = sum(1 for r in cust_rows if r[CUSTOMER_COLS.index("age")] is None)
    print(f"customers with null age: {n_null_age}")
    if n_null_age == 0:
        raise SystemExit("seeds must contain a null age")

    art_rows = [
        tuple(r[c] for c in ARTICLE_COLS)
        for r in (
            spark.table(ingest.ARTICLES_TABLE)
            .filter(F.col("article_id").isin(kept_articles))
            .orderBy("article_id")
            .collect()
        )
    ]

    _write(SEED_DIR / "seed_transactions.csv", TRANSACTION_COLS, txn_rows)
    _write(SEED_DIR / "seed_customers.csv", CUSTOMER_COLS, cust_rows)
    _write(SEED_DIR / "seed_articles.csv", ARTICLE_COLS, art_rows)
    spark.stop()


if __name__ == "__main__":
    main()
