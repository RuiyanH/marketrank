from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.types import (
    DateType, DoubleType, IntegerType, StringType, StructField, StructType,
)

from marketrank import config

# how to read the csv files
CUSTOMERS_SCHEMA = StructType([
    StructField("customer_id",            StringType(),  nullable=False),
    StructField("FN",                     DoubleType(),  nullable=True),
    StructField("Active",                 DoubleType(),  nullable=True),
    StructField("club_member_status",     StringType(),  nullable=True),
    StructField("fashion_news_frequency", StringType(),  nullable=True),
    StructField("age",                    IntegerType(), nullable=True),
    StructField("postal_code",            StringType(),  nullable=True),
])

CUSTOMERS_TABLE = f"{config.CATALOG}.raw.customers"

TRANSACTIONS_SCHEMA = StructType([
    StructField("t_dat", DateType(), nullable=False),
    StructField("customer_id", StringType(), nullable=False),
    StructField("article_id", StringType(), nullable=False),
    StructField("price", DoubleType(), nullable=True),
    StructField("sales_channel_id", IntegerType(), nullable=True),
])

TRANSACTIONS_TABLE = f"{config.CATALOG}.raw.transactions"

ARTICLES_SCHEMA = StructType([
    StructField("article_id",                   StringType(),  nullable=False),
    StructField("product_code",                 StringType(),  nullable=True),
    StructField("prod_name",                    StringType(),  nullable=True),
    StructField("product_type_no",              IntegerType(), nullable=True),
    StructField("product_type_name",            StringType(),  nullable=True),
    StructField("product_group_name",           StringType(),  nullable=True),
    StructField("graphical_appearance_no",      IntegerType(), nullable=True),
    StructField("graphical_appearance_name",    StringType(),  nullable=True),
    StructField("colour_group_code",            StringType(),  nullable=True),
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

def create_tables(spark: SparkSession) -> None:
    """
    Create the namespace and raw tables: tells Spark how to store the data.
    Safe to re-run.
    """
    spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {config.CATALOG}.raw")

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {TRANSACTIONS_TABLE} (
            t_dat             DATE,
            customer_id       STRING,
            article_id        STRING,
            price             DOUBLE,
            sales_channel_id  INT, 
            _ingested_at      TIMESTAMP
        )
        USING iceberg
        PARTITIONED BY (days(t_dat))
    """)

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {CUSTOMERS_TABLE} (
            customer_id             STRING,
            FN                      DOUBLE,
            Active                  DOUBLE,
            club_member_status      STRING,
            fashion_news_frequency  STRING,
            age                     INT,
            postal_code             STRING
        )
        USING iceberg
    """)

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {ARTICLES_TABLE} (
            article_id                   STRING,
            product_code                 STRING,
            prod_name                    STRING,
            product_type_no              INT,
            product_type_name            STRING,
            product_group_name           STRING,
            graphical_appearance_no      INT,
            graphical_appearance_name    STRING,
            colour_group_code            STRING,
            colour_group_name            STRING,
            perceived_colour_value_id    INT,
            perceived_colour_value_name  STRING,
            perceived_colour_master_id   INT,
            perceived_colour_master_name STRING,
            department_no                INT,
            department_name              STRING,
            index_code                   STRING,
            index_name                   STRING,
            index_group_no               INT,
            index_group_name             STRING,
            section_no                   INT,
            section_name                 STRING,
            garment_group_no             INT,
            garment_group_name           STRING,
            detail_desc                  STRING
        )
        USING iceberg
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

def read_customers_csv(spark: SparkSession) -> DataFrame:
    """Read the raw customers CSV with a declared schema."""
    return (
        spark.read
        .schema(CUSTOMERS_SCHEMA)
        .option("header", True)
        .csv(str(config.DATA_RAW / "customers.csv"))
    )

def read_articles_csv(spark: SparkSession) -> DataFrame:
    """Read the raw articles CSV with a declared schema."""
    return (
        spark.read
        .schema(ARTICLES_SCHEMA)
        .option("header", True)
        .csv(str(config.DATA_RAW / "articles.csv"))
    )

def load_transactions(spark: SparkSession, start=None, end=None) -> None:
    """
    Replace whole days in the target table. Safe to re-run.
    Full-day refresh, not a merge: every day present in `df` is replaced wholesale. 
    """
    df = read_transactions_csv(spark, start=start, end=end)
    df = df.withColumn("_ingested_at", F.current_timestamp())
    df.writeTo(TRANSACTIONS_TABLE).overwritePartitions()

def load_customers(spark: SparkSession) -> None:
    """Merge the customers file into the raw table on customer_id. Safe to re-run."""
    read_customers_csv(spark).createOrReplaceTempView("customers_source")
    spark.sql(f"""
        MERGE INTO {CUSTOMERS_TABLE} AS t
        USING customers_source AS s
        ON t.customer_id = s.customer_id
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)

def load_articles(spark: SparkSession) -> None:
    """Merge the articles file into the raw table on article_id. Safe to re-run."""
    read_articles_csv(spark).createOrReplaceTempView("articles_source")
    spark.sql(f"""
        MERGE INTO {ARTICLES_TABLE} AS t
        USING articles_source AS s
        ON t.article_id = s.article_id
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)



