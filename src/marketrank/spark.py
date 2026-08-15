from pyspark.sql import SparkSession

from marketrank import config


def get_spark(
    app_name: str = "marketrank",
    driver_memory: str = "8g",
    master: str = "local[*]",
) -> SparkSession:
    """
    Build a Spark session.

    The Iceberg jar coordinate, the SQL extensions, the `local` catalog and the
    session timezone are NOT set here -- they live in conf/spark-defaults.conf,
    which dbt-spark's session method also reads (it builds its own SparkSession
    and cannot be handed them afterwards). config.py sets SPARK_CONF_DIR at
    import time, which is before this function runs.

    What stays here is runtime and topology configuration: it is machine- and
    mode-dependent and a static conf file cannot branch on it.
    """
    # Importing config is what sets SPARK_CONF_DIR and renders the conf file;
    # touching it here makes the dependency explicit rather than incidental.
    config.render_spark_defaults()

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_memory)
    )

    if master.startswith("local"):
        # Our hostname resolves to a stale DHCP address, so Spark's default
        # bind target may not exist. In local mode the driver only talks to
        # threads inside its own JVM, so loopback is the correct target.
        # This MUST stay conditional: a multi-node standalone cluster breaks if
        # executors on other nodes try to reach a driver bound to 127.0.0.1.
        builder = (
            builder
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.driver.host", "127.0.0.1")
        )

    return builder.getOrCreate()
