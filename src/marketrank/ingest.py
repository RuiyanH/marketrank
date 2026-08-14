from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import (
    DateType, DoubleType, IntegerType, StringType, StructField, StructType,
)

from marketrank import config

TRANSACTIONS_SCHEMA = StructType([
    StructField("t_dat", DateType(), nullable=False),
    StructField("customer_id", StringType(), nullable=False),
    StructField("article_id", StringType(), nullable=False),
    StructField("price", DoubleType(), nullable=True),
    StructField("sales_channel_id", IntegerType(), nullable=True),
])

TRANSACTIONS_TABLE = f"{config.CATALOG}.raw.transactions"

# article_id is a ZERO-PADDED STRING ("0663713001"). Inferred as a long the
# leading zero is lost and it stops joining to transactions. Declared, never
# inferred — same rule as transactions.
ARTICLES_SCHEMA = StructType([
    StructField("article_id",                   StringType(),  nullable=False),
    StructField("product_code",                 StringType(),  nullable=True),
    StructField("prod_name",                    StringType(),  nullable=True),
    StructField("product_type_no",              IntegerType(), nullable=True),
    StructField("product_type_name",            StringType(),  nullable=True),
    StructField("product_group_name",           StringType(),  nullable=True),
    StructField("graphical_appearance_no",      IntegerType(), nullable=True),
    StructField("graphical_appearance_name",    StringType(),  nullable=True),
    StructField("colour_group_code",            IntegerType(), nullable=True),
    StructField("colour_group_name",            StringType(),  nullable=True),
    StructField("perceived_colour_value_id",    IntegerType(), nullable=True),
    StructField("perceived_colour_value_name",  StringType(),  nullable=True),
    StructField("perceived_colour_master_id",   IntegerType(), nullable=True),
    StructField("perceived_colour_master_name", StringType(),  nullable=True),
    StructField("department_no",                IntegerType(), nullable=True),
    StructField("department_name",              StringType(),  nullable=True),
    StructField("index_code",                   StringType(),  nullable=True),
    StructField("index_name",                   StringType(),  nullable=True),
    StructField("index_group_no",               IntegerType(), nullable=True),
    StructField("index_group_name",             StringType(),  nullable=True),
    StructField("section_no",                   IntegerType(), nullable=True),
    StructField("section_name",                 StringType(),  nullable=True),
    StructField("garment_group_no",             IntegerType(), nullable=True),
    StructField("garment_group_name",           StringType(),  nullable=True),
    StructField("detail_desc",                  StringType(),  nullable=True),
])

ARTICLES_TABLE = f"{config.CATALOG}.raw.articles"

# FN and Active are sparse 1.0/null floats and age has nulls. All three stay
# nullable and uncast here — the cast happens at the staging layer, not at read,
# so that raw is a faithful copy of what the producer sent.
CUSTOMERS_SCHEMA = StructType([
    StructField("customer_id",            StringType(), nullable=False),
    StructField("FN",                     DoubleType(), nullable=True),
    StructField("Active",                 DoubleType(), nullable=True),
    StructField("club_member_status",     StringType(), nullable=True),
    StructField("fashion_news_frequency", StringType(), nullable=True),
    StructField("age",                    IntegerType(), nullable=True),
    StructField("postal_code",            StringType(), nullable=True),
])

CUSTOMERS_TABLE = f"{config.CATALOG}.raw.customers"


def create_tables(spark: SparkSession) -> None:
    """Create the namespace and raw tables. Safe to re-run."""
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {config.CATALOG}.raw")
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {TRANSACTIONS_TABLE} (
            t_dat             DATE,
            customer_id       STRING,
            article_id        STRING,
            price             DOUBLE,
            sales_channel_id  INT
        )
        USING iceberg
        PARTITIONED BY (days(t_dat))
    """)

def read_transactions_csv(spark: SparkSession, start=None, end=None) -> DataFrame:
    """Read the raw CSV with a declared schema, optionally limited to a date range."""
    df = (
        spark.read
        .schema(TRANSACTIONS_SCHEMA)
        .option("header", True)
        .csv(str(config.DATA_RAW / "transactions_train.csv"))
    )
    if start is not None:
        df = df.filter(F.col("t_dat") >= F.lit(start))
    if end is not None:
        df = df.filter(F.col("t_dat") <= F.lit(end))
    return df


def read_articles_csv(spark: SparkSession) -> DataFrame:
    """Read articles.csv with a declared schema."""
    return (
        spark.read
        .schema(ARTICLES_SCHEMA)
        .option("header", True)
        .option("quote", '"')
        .option("escape", '"')
        .csv(str(config.DATA_RAW / "articles.csv"))
    )


def read_customers_csv(spark: SparkSession) -> DataFrame:
    """Read customers.csv with a declared schema."""
    return (
        spark.read
        .schema(CUSTOMERS_SCHEMA)
        .option("header", True)
        .option("quote", '"')
        .option("escape", '"')
        .csv(str(config.DATA_RAW / "customers.csv"))
    )


def load_transactions(spark: SparkSession, start=None, end=None) -> None:
    """
    Replace whole days in the target table. Safe to re-run.

    This is a FULL-DAY REFRESH from the source of truth, not a merge:
    overwritePartitions() replaces every partition present in the incoming
    DataFrame wholesale, so feeding it a delta for a day DELETES that day's
    other rows. Applying deltas (late-arriving data) is week 9's job and needs
    a different mechanism -- see docs/IMPLEMENTATION.md step 9.1.
    """
    df = read_transactions_csv(spark, start=start, end=end)
    df.writeTo(TRANSACTIONS_TABLE).overwritePartitions()


def load_articles(spark: SparkSession) -> None:
    """
    Replace local.raw.articles wholesale from the CSV snapshot.

    createOrReplace(), not partition overwrite: articles.csv is a snapshot of
    *current state*, not an event log, and a dimension has no meaningful
    partition. The write strategy follows the grain, not preference.
    createOrReplace also owns the schema, so there is no separate CREATE TABLE
    DDL that can drift out of step with the live table.
    """
    read_articles_csv(spark).writeTo(ARTICLES_TABLE).createOrReplace()


def load_customers(spark: SparkSession) -> None:
    """
    Replace local.raw.customers wholesale from the CSV snapshot.

    Same reasoning as load_articles: a current-state snapshot, unpartitioned,
    replaced whole.
    """
    read_customers_csv(spark).writeTo(CUSTOMERS_TABLE).createOrReplace()




