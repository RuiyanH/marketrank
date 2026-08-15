# Ordered, re-runnable jobs. No scheduler -- nothing here arrives on a
# schedule, so a Makefile is the correct amount of orchestration.
PY := .venv/bin/python
DBT := .venv/bin/dbt

# dbt never imports marketrank, so it never runs config.py's
# os.environ.setdefault("SPARK_CONF_DIR", ...). It needs the export.
# `render-conf` is what writes the file that export points at.
export SPARK_CONF_DIR := $(CURDIR)/conf
# profiles.yml is committed next to dbt_project.yml, not in ~/.dbt
export DBT_PROFILES_DIR := $(CURDIR)/dbt

# The loopback bind workaround, for the one consumer that cannot get it from
# get_spark(): dbt-spark's session method builds its own SparkSession and never
# calls us. It goes here rather than in conf/spark-defaults.conf because it is
# machine-specific topology config -- this laptop's hostname resolves to a stale
# DHCP address -- and a committed conf file would break a multi-node cluster.
# SPARK_LOCAL_IP is the environment-level spelling of
# spark.driver.bindAddress/host, so the split by kind survives intact.
export SPARK_LOCAL_IP := 127.0.0.1

# macOS only: the lightgbm wheel links @rpath/libomp.dylib and searches only
# Homebrew/MacPorts prefixes, so it fails to load unless libomp is installed
# system-wide. torch ships one. DYLD_* is read by dyld at process start, so this
# cannot be done from config.py -- it has to be in the environment.
TORCH_LIB := $(CURDIR)/.venv/lib/python3.11/site-packages/torch/lib
export DYLD_LIBRARY_PATH := $(TORCH_LIB)

.PHONY: help render-conf load-raw seeds dbt dbt-ci features twotower-data candidates test test-fast clean-dbt

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n", $$1, $$2}'

render-conf: ## Render conf/spark-defaults.conf from its template
	$(PY) -m marketrank.config

load-raw: render-conf ## Load all three raw Iceberg tables from the CSVs
	$(PY) -m marketrank.ingest

seeds: render-conf ## Regenerate dbt's committed seed CSVs from the warehouse
	$(PY) -m marketrank.make_seeds

dbt: render-conf ## Build the dimensional model on Spark/Iceberg
	cd dbt && $(CURDIR)/$(DBT) build --target dev

dbt-ci: ## Build the same models and tests on DuckDB over the seeds
	cd dbt && $(CURDIR)/$(DBT) build --target ci

features: render-conf ## Build/backfill the PIT feature tables
	$(PY) -m marketrank.features $(START) $(END)

twotower-data: render-conf ## Export the two-tower training set to artifacts/
	$(PY) -m marketrank.retrieval.dataset $(COHORT)

test: render-conf ## Full pytest suite, Spark tests included
	.venv/bin/pytest -q

test-fast: ## pytest without the Spark tests
	.venv/bin/pytest -q -m "not spark"

clean-dbt:
	rm -rf dbt/target dbt/logs
