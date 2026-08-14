from pyspark.sql import SparkSession

from marketrank import config

ICEBERG_PACKAGE = (
    f"org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:{config.ICEBERG_VERSION}"
)


def get_spark(app_name: str = "marketrank", driver_memory: str = "8g", master: str = "local[*]") -> SparkSession:
    """Building an Iceberg-aware Spark session."""
    cat = config.CATALOG
    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.jars.packages", ICEBERG_PACKAGE)
        .config(
            "spark.sql.extensions",
            "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions",
        )
        .config(f"spark.sql.catalog.{cat}", "org.apache.iceberg.spark.SparkCatalog")
        .config(f"spark.sql.catalog.{cat}.type", "hadoop")
        .config(f"spark.sql.catalog.{cat}.warehouse", str(config.WAREHOUSE))
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.session.timeZone", "UTC")
    )

    if master.startswith("local"):
        # Our hostname resolves to a stale DHCP address, so Spark's default
        # bind target may not exist. In local mode the driver only talks to
        # threads inside its own JVM, so loopback is the correct target.
        builder = (
            builder
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.driver.host", "127.0.0.1")
        )

    return builder.getOrCreate()