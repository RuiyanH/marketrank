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


def load_transactions(spark: SparkSession, start=None, end=None) -> None:
    """Replace whole days in the target table. Safe to re-run."""
    df = read_transactions_csv(spark, start=start, end=end)
    df.writeTo(TRANSACTIONS_TABLE).overwritePartitions()




