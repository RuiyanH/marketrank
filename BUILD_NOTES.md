# BUILD_NOTES.md

Build log for `docs/IMPLEMENTATION.md`, executed on branch `build/implementation`
in an isolated worktree at `/Users/test/Developer/marketrank-build`.

Every number below was produced by a command that actually ran. Where a number
was not produced, the entry says **not run** rather than an estimate.

## Environment this was built on

| Fact | Value |
|---|---|
| Machine | Apple Silicon macOS 25.5.0, **8 cores, 16 GiB RAM** |
| Free disk at start | **15 GiB** on the volume holding the worktree |
| Python | 3.11.15 (`/opt/homebrew/bin/python3.11` → worktree `.venv`) |
| Java | Temurin 17.0.20 |
| `MARKETRANK_DATA_RAW` | `/Users/test/Developer/marketrank/data/raw` (read-only) |
| `MARKETRANK_WAREHOUSE` | `/Users/test/Developer/marketrank-build/warehouse` (built from scratch) |

**The single most important fact about this build:** 8 cores / 16 GiB / 15 GiB
free disk is roughly two orders of magnitude below what week 4's candidate
generation needs (§5 of the spec sizes it at 40–160 GB of *output*). Everything
through week 2 ran at full scale. From week 3 on, full-scale code was written and
a **reduced configuration** was executed; each reduction is stated at the step and
carried into every number derived from it.

---

## Step 1.0 — Declare your dependencies

**Checkpoint:** fresh venv + `pip install -e ".[dev]"` gives working `pytest` and `dbt`.

Observed, on a venv created from scratch with `python3.11 -m venv .venv`:

```
pytest 9.1.1
dbt Core: 1.12.2   plugins: duckdb 1.11.0, spark 1.11.0
```

Resolved versions worth recording (they are not pinned beyond `pyspark`):
pyspark 3.5.9, dbt-core 1.12.2, dbt-spark 1.11.0, dbt-duckdb 1.11.0, duckdb 1.5.5,
py4j 0.10.9.9. **PASS.**

Deviation: `lightgbm`, `torch`, `hnswlib` are *not* in the `dev` extra at this
commit. The doc marks them "(later)", so they are added in the commit for the week
that first needs them, to keep the git history honest about when each arrived.

## Step 1.1 — Load `articles` and `customers`

**Checkpoint (marked VERIFY in the doc):** `local.raw.articles` = 105,542 and
`local.raw.customers` = 1,371,980.

Observed, on a warehouse built from scratch:

```
articles count:   105542      (load wall clock 4.0 s)
customers count: 1371980      (load wall clock 2.8 s)
```

**VERIFY resolved: the doc's widely-cited counts are correct.** They also match
`wc -l` on the CSVs minus the header (105,543 / 1,371,981 lines), which is the
independent check worth doing — it also proves no field in `articles.csv`
contains an embedded newline, so no `multiLine` read option is needed.

So §2's résumé claim should read **105,542 articles / 1,371,980 customers**.

Zero-padding check: `article_id` reads back as `0108775015`, string, leading zero
intact.

Deviation: `create_tables()` declares DDL only for `raw.transactions`. `articles`
and `customers` are created by `createOrReplace()` in their loaders, which owns
their schema. That is deliberate — §0's problem #1 is a table and a `CREATE TABLE`
statement drifting apart, and the cheapest way to not have that problem for a
snapshot-replaced table is to not have a second declaration of its schema.

## Step 1.2 — Decide the write strategy, deliberately

No checkpoint; the step is a comment and a decision.

**Think-first answer, confirmed by reading the code path rather than guessing:**
`writeTo(...).overwritePartitions()` is Spark's dynamic partition overwrite, so a
corrections file containing 3 rows for `2019-06-01` leaves exactly 3 rows in that
partition. The doc is right. Nothing to run here — the same fact is *demonstrated*
in step 9.1, and that is where the number goes.

Recorded so week 9 does not have to re-derive it: the mechanism the merge case
needs is a key, and the raw transaction rows have none. That is what step 1.4 is
for.

## Step 1.3 — `_ingested_at`, using schema evolution for real

Note on §0's problem #1 (`promo_flag`): it does not exist in this build. The
divergence between `create_tables()` and the live table came from the warehouse
having been evolved by hand; a warehouse rebuilt from the committed DDL simply
does not have the column. So `_ingested_at` is the *first* schema evolution here,
and it is the one with a reason. If you are redoing this on the existing
warehouse, drop `promo_flag` in the same commit.

**Base load, for the record** (fresh warehouse, full `transactions_train.csv`):

```
TXN_LOAD_SECONDS 49.3
TXN_COUNT        31,788,324
TXN_SPAN         2018-09-20 .. 2020-09-22   DAYS 734
warehouse size after load: 559 MB
```

31.79M is the number to use wherever the spec writes "~32M".

### VERIFY — what `overwritePartitions()` does when the DataFrame lacks a column

The doc asks whether Iceberg fills null or refuses. **It refuses**, at analysis
time, before any work:

```
AnalysisException [INCOMPATIBLE_DATA_FOR_TABLE.CANNOT_FIND_DATA]
Cannot write incompatible data for the table `local`.`raw`.`transactions`:
Cannot find data for the output column `_ingested_at`.
```

Worth stating as two directions rather than one rule, because they differ:

- **Read-side drift is tolerated.** Data files written before the column existed
  read back as null — that is the metadata-only evolution working.
- **Write-side drift is rejected.** `DataFrameWriterV2` resolves the incoming
  DataFrame against the *current* table schema by name and fails on a missing
  column. Producer and table drifting apart is a loud failure at write, not a
  silent column of nulls.

That asymmetry is the actual answer to "what happens when the producer and the
table drift apart", and it is a better one than the doc's either/or implies.

### Evidence the evolution was free

`transactions.snapshots` after the ALTER contains **3** snapshots — the initial
load and the two step-1.3 day reloads. The ALTER produced a
`metadata_log_entries` row at `23:11:39.998` and **no snapshot at all**, and
`latest_schema_id` went 0 → 1 only at the next write. No data files were touched:

```
snapshot_id          committed_at              operation  added      total
2968744110833297794  2026-08-14 23:10:28.944   overwrite  31788324   31788324
4827918704589656546  2026-08-14 23:12:39.522   overwrite  62619      31788324
5931284054021401369  2026-08-14 23:12:56.625   overwrite  62619      31788324
```

### Checkpoint — both facts, together

Day used: `2019-06-01` (62,619 rows), loaded twice back to back.

```
distinct _ingested_at, run a: 2026-08-14 19:12:21.582137
distinct _ingested_at, run b: 2026-08-14 19:12:41.626972
assert_identical(a, b)                -> ignored ['_ingested_at']; 0 / 0; IDENTICAL
assert_identical(a, b, ignore_cols=()) -> 62619 / 62619; AssertionError
rows still NULL in _ingested_at elsewhere: 31,725,705 of 31,788,324
```

31,725,705 + 62,619 = 31,788,324, so exactly the reloaded day carries a
timestamp. **PASS**, both halves.

### Deviation — `assert_identical`'s default

The doc specifies `assert_identical(a, b, ignore_cols=("_ingested_at",))`, but
its own prose asks for "anything else you later add with a leading underscore"
and for a rule rather than a hardcoded column name. Those two are in tension:
week 9 adds `_source_file`, and with the literal default in place `assert_identical`
would go red the first time a corrections batch lands, for a reason that has
nothing to do with idempotency.

Implemented as `ignore_cols=None` meaning *every column whose name starts with
`_`*, with an explicit tuple accepted as an override and `()` demanding a
whole-row match. `()` is what the second half of this checkpoint uses, so the
override is not speculative API — the checkpoint needs it.

## Step 1.4 — Resolve the grain

Measured on the full log before deciding (all four numbers from one Spark job):

```
source rows                                              31,788,324
distinct (customer, article, t_dat, channel)             28,583,889
distinct (customer, article, t_dat, channel, price)      28,813,419
groups with >1 distinct price                               223,068  (0.78%)
max distinct prices inside one group                              8
distinct price values                                         9,857
distinct price values cast to DECIMAL(10,8)                   9,857
price range                          1.694915254237288e-05 .. 0.5915254237288136
```

**Chosen: option (b), price OUT of the key.** Grain
`(customer_id, article_id, t_dat, sales_channel_id)`, 28,583,889 rows.

**Multi-quantity purchase rate = 10.08%** (3,204,435 of 31,788,324 source rows
collapse). With price kept in the key it would have been 9.36% — the gap between
those two, 229,530 rows, is the mid-day-markdown population.

Reasoning recorded because the doc leaves this decision open: a markdown does not
make a second basket line, and carving 0.78% of groups into extra rows to
preserve a price difference is a modeling artifact rather than a business
distinction. `fact_transaction` instead carries `price_mean = sum(price)/qty`,
`price_min`, `price_max` (so the markdown stays visible) and
`revenue = sum(price)`. Note `revenue = sum(price)` is exact and identical to
`qty * price_mean`, so nothing is lost even in the multi-price groups — this is
strictly better than the doc's "accept the loss" framing of the mean-price
variant.

**On the `DECIMAL(10,8)` advice:** with price out of the key it stops mattering,
exactly as the doc predicts. Measured anyway, since it is cheap and the doc gives
no number: all 9,857 distinct prices survive the cast without collision, so the
`DECIMAL(10,8)` recommendation would have been safe. Worth knowing that 8 decimal
places is *not* obviously enough by inspection — the raw values carry ~18
significant digits (`0.050830508474576264`) — and the reason it works is that the
underlying price grid is coarse: 9,857 values over a 4-decade range.

**Checkpoint — measured half.** The count query ran and is quoted above.
`count(distinct customer_id, article_id, t_dat, sales_channel_id)` = 28,583,889,
which is by construction the row count a `GROUP BY` on those four columns
produces, so the grain is unique. Source minus fact = 31,788,324 − 28,583,889 =
**3,204,435 rows**, the multi-quantity purchase population.

**Checkpoint — not yet discharged as a property of a built table.** At this commit
there is no `dbt/` directory and no materialized `fact_transaction`. The number
above is a property of a query over `raw.transactions`, not of a table that
exists. Step 1.5 below records the materialized row count and the dbt uniqueness
test result when they have actually been run.

## Step 1.5 — Stand up dbt on Spark

This step needed four corrections to the doc, three of them because the doc's
design is right but incomplete about *which* consumer gets what. All four are
below with the exact error each produced, because each one costs an hour to
diagnose from scratch.

### Correction 1 — a static `spark-defaults.conf` cannot carry the warehouse path

The doc says to create `conf/spark-defaults.conf` holding, among other things,
"the `local` catalog config". The catalog config includes
`spark.sql.catalog.local.warehouse`, and the warehouse location is
environment-dependent by design: `config.py` reads `MARKETRANK_WAREHOUSE`, and
`SETUP_MISHA.md`'s `env.misha.sh` sets a different one. **Spark does not
interpolate environment variables in `spark-defaults.conf`.** Tested directly:

```
spark.sql.catalog.local.warehouse   ${env:MARKETRANK_WAREHOUSE}

-> s.conf.get(...) returns the literal string "${env:MARKETRANK_WAREHOUSE}"
-> IllegalArgumentException: java.net.URISyntaxException:
   Relative path in absolute URI: ${env:MARKETRANK_WAREHOUSE%7D
```

So a committed static file would have to hardcode one machine's path, which
contradicts the runbook it is supposed to serve.

**Resolution:** `conf/spark-defaults.conf.template` is committed;
`marketrank.config.render_spark_defaults()` renders `conf/spark-defaults.conf`
from it at import time (idempotent, rewrites only on change), and the rendered
file is gitignored. `make render-conf` / `python -m marketrank.config` is the
entry point for the one consumer that does not import `marketrank` — dbt. This
keeps the doc's actual goal (one definition of the catalog, visible to both
consumers) while letting the warehouse move per machine.

### Correction 2 — dbt-spark's session has no driver bind address, and dies

The doc's split-by-kind rule puts `spark.driver.bindAddress` /
`spark.driver.host` in `get_spark()` because they are machine- and mode-
dependent. Correct — but dbt-spark's session method never calls `get_spark()`,
so on this laptop (hostname resolves to a stale DHCP address) `dbt build` failed
before running a single model:

```
WARN Utils: Service 'sparkDriver' could not bind on a random free port.
  ... io.netty.channel.AbstractChannel.bind ... make: *** [dbt] Error 2
```

**Resolution that preserves the split:** export `SPARK_LOCAL_IP=127.0.0.1` from
the Makefile. It is the environment-level spelling of the same two settings, so
it stays out of the committed conf file (where it would break a multi-node
cluster) and still reaches dbt. The doc should say this; without it, step 1.5's
checkpoint is unreachable on any machine that needs the loopback workaround —
which is the machine the doc was written on.

### Correction 3 — Iceberg has no views, so staging cannot be views

The doc says staging models are views. On the Spark target every one of them
failed:

```
ERROR creating sql view model staging.stg_articles
  Replacing a view is not supported by catalog: local
```

Iceberg's `SparkCatalog` (1.11.0, Hadoop catalog) does not implement
`ViewCatalog`. **Resolution: `ephemeral`.** Staging here is pure projection
(rename + cast, no business logic), so inlining it as a CTE stores nothing and
costs nothing. It is also better than the obvious alternative of "table on
Spark, view on DuckDB", because that would make dev and CI materialise
differently — the opposite of what step 1.6 wants from CI.

### Correction 4 — seeds cannot build on the Spark target

```
ERROR loading seed file raw_seed.seed_articles
  [DATATYPE_MISSING_SIZE] DataType "VARCHAR" requires a length parameter
```

dbt-spark emits unsized `varchar` for seed columns typed as strings, which Spark
rejects. Seeds exist so CI can run without the 3.5 GB extract, so they have no
business on the Spark target at all. **Resolution:**
`+enabled: "{{ target.type == 'duckdb' }}"` in `dbt_project.yml`.

### VERIFY — does `spark.sql.defaultCatalog=local` do the routing?

**Yes.** With it set and no catalog qualification anywhere in `dbt_project.yml`
or `sources.yml`, dbt's unqualified `marts.fact_transaction` landed in the
Iceberg catalog:

```
SHOW NAMESPACES IN local  ->  staging, marts, raw_seed, raw
```

The fallback the doc names (fully qualifying the catalog in `dbt_project.yml`)
was not needed. Note the flip side: because the default catalog is Iceberg,
*everything* dbt does goes there, which is exactly how correction 3 surfaced.

### Checkpoint

`make dbt` (Spark/Iceberg target), from the real 31.8M-row warehouse:

```
1 of 16 OK created sql table model marts.dim_article        OK in  4.64s
2 of 16 OK created sql table model marts.dim_customer       OK in  3.38s
3 of 16 OK created sql table model marts.fact_transaction   OK in 59.39s
... 13 data tests ...
Done. PASS=16 WARN=0 ERROR=0 SKIP=0 NO-OP=0 REUSED=0 TOTAL=16
wall clock: 2 min 09 s
```

Is it genuinely Iceberg?

```
SELECT * FROM local.marts.fact_transaction.snapshots
  7609064981163123657  overwrite  total-records = 28583889

SHOW TBLPROPERTIES local.marts.fact_transaction
  format          iceberg/parquet
  format-version  2
  write.parquet.compression-codec  zstd
```

The `.snapshots` metadata table only resolves for a real Iceberg table. **PASS.**

**This also discharges step 1.4's pending half:** `fact_transaction` materialised
at **28,583,889 rows**, exactly the distinct-key count measured in 1.4, and
`assert_fact_transaction_grain_unique` passed on the full table (31.37 s).

Slowest tests on the real warehouse, for anyone budgeting the run:
grain uniqueness 31.37 s, customer relationships 14.24 s, article relationships
2.79 s. All referential-integrity tests pass, so every transaction's
`article_id` and `customer_id` really is present in the dimension files.

## Step 1.6 — Tests, and the second target

Seeds are generated by `src/marketrank/make_seeds.py` (committed, deterministic,
re-runnable via `make seeds`) rather than typed by hand, so they are real rows
with real edge cases:

```
seed_transactions.csv  355 rows
seed_customers.csv      50 rows
seed_articles.csv       47 rows
multi-quantity duplicate rows: 158
customers with null age: 1
```

Doc asked for ~200 / ~50 / ~50; 355 transactions is what falls out of taking 49
customers with 3–8 purchases each, and cutting to exactly 200 would have dropped
half the customers. **Read the seed's 44% multi-quantity rate as a fixture
property, not a dataset property** — it is a selection artifact of restricting to
the 50 most popular articles, and the real rate is 10.08% (step 1.4).

`tests/test_iceberg.py` builds its own five-line CSV in `tmp_path` and
monkeypatches `config.DATA_RAW`, so it drives the *real* write path
(declared-schema CSV read → `_ingested_at` stamp → `overwritePartitions`) with no
dependency on the 3.5 GB extract. That is what lets it run in CI. `ingest` gained
a `table=` parameter for the same reason — the test writes to a throwaway table
in a `test_tmp` namespace and drops it afterwards, rather than touching the
warehouse.

Three tests, not one:

1. `test_reload_is_idempotent` — the doc's test, driving two loads.
2. `test_whole_row_diff_still_sees_the_metadata_column` — asserts
   `assert_identical(..., ignore_cols=())` *raises*. Without it, the first test
   would keep passing if `_ingested_at` ever stopped varying, and would then be
   proving nothing. The step-1.3 checkpoint makes this pairing explicit, so it
   belongs in the suite rather than in a shell transcript.
3. `test_partition_overwrite_replaces_a_day_wholesale` — step 1.2's Think-first
   answer as an executable claim (load 3 rows, then load a 1-row "corrections"
   file for the same day, get 1 row). Week 9 has to change this behaviour, and
   this is the test that will go red when it does — deliberately.

**Checkpoint.**

```
make dbt-ci   ->  Done. PASS=19 WARN=0 ERROR=0 SKIP=0 TOTAL=19   in 4.5 s wall
pytest -m spark ->  3 passed in 11.05 s
```

Under a minute, with `MARKETRANK_DATA_RAW` never read. **PASS.**

## Step 1.7 — GitHub Actions

`.github/workflows/ci.yml`: `pull_request` → checkout → setup-python 3.11 →
setup-java 17 → `pip install -e ".[dev]"` → `dbt build --target ci` →
`pytest -v`. The Spark tests run (the doc's recommended option), because gate 1's
criterion is the PIT test and a job that skips it is not enforcing the gate.

**Deviation, and it is a real gap in this build: the workflow has never
executed.** The task brief forbids pushing to any remote, so no PR was opened,
no runner ran, and there is no green or red check anywhere. What was actually
done instead:

- The workflow's command sequence was run locally in the same order, from a
  shell where `MARKETRANK_DATA_RAW` was set but unused by either step.
- **"Watch it fail once"** was performed locally. `fact_transaction` was
  rewritten to grain (a) — one row per source row, the mistake step 1.4 rejects
  — and the CI target went red on the grain test:

  ```
  12 of 19 FAIL 98 assert_fact_transaction_grain_unique
           Got 98 results, configured to fail if != 0
  Done. PASS=18 WARN=0 ERROR=1 SKIP=0 TOTAL=19
  make exit status: 2
  ```

  Restoring the model returned it to `PASS=19 ERROR=0`. So the check *would* be
  red, and it is red for the right reason.
- **"The Spark tests actually ran rather than being collected and skipped"** was
  checked with `pytest -v`:

  ```
  tests/test_iceberg.py::test_reload_is_idempotent PASSED
  tests/test_iceberg.py::test_whole_row_diff_still_sees_the_metadata_column PASSED
  tests/test_iceberg.py::test_partition_overwrite_replaces_a_day_wholesale PASSED
  3 passed
  ```

  No SKIPPED lines. Note this only proves it on macOS with a Temurin 17 already
  installed; the `setup-java` step is unverified.

**Not run: anything involving GitHub.** No push, no PR, no badge. If you are
following these notes, this is the one step you have to do yourself, and the
first push is where `setup-java` and the runner's `pip install -e ".[dev]"` get
their first real test.

**PR discipline** is simulated with local branches: this week's steps 1.5–1.7
were committed on `feature/week1-dbt-ci` and merged into `build/implementation`
with `--no-ff`, so the merge structure exists in the history even though no PR
does.

---

## Step 2.0 — The dataset fact that shapes everything

Nothing to build. Both consequences are written into the README's limitations
list: the `[d − w, d − 1]` rule (date-only `t_dat`, so same-day events cannot be
ordered and are excluded entirely) and the current-state-snapshot dimensions.

## Step 2.1 — Write the leakage test first, and watch it fail

**Checkpoint: both tests fail with a clear message, committed failing.**

```
E  ImportError: cannot import name 'features' from 'marketrank'
   tests/test_pit.py:69: ImportError
E  ImportError: cannot import name 'features' from 'marketrank'
   tests/test_pit.py:90: ImportError
FAILED tests/test_pit.py::test_window_excludes_same_day
FAILED tests/test_pit.py::test_future_events_do_not_change_past_features
2 failed in 4.03s
```

**PASS** (in the sense the step means: they fail, for the right reason, and the
commit that adds them precedes the commit that adds `features.py`).

Implementation note: `from marketrank import features` is inside each test body,
not at module scope. At module scope the import error is a *collection* error —
pytest reports the file as broken rather than the two tests as failing, which is
a worse artifact for the history the step is trying to create.

**Deviation, and it is forced by this very step.** The doc specifies
`daily_customer_agg(spark)` in step 2.2 — a function that takes a session and
reads the warehouse. A function shaped like that cannot be driven by "a tiny
hand-built DataFrame", which is what step 2.1 requires. The two steps contradict
each other. Resolved in favour of 2.1: the aggregate and window functions take
**DataFrames**, and a separate `build_features(spark, ...)` is the only thing
that touches tables. That is also the shape that makes the backfill parameter of
step 2.4 natural, so the doc's own later step wants it too.

## Steps 2.2 / 2.3 — daily aggregates and the rolling windows

**Deviation on the source table.** The feature pipeline reads
`local.raw.transactions`, not `local.marts.fact_transaction`. With raw,
`n_txn = count(*)` counts units purchased and `spend = sum(price)` is exact,
which is what the aggregate functions are written for; reading the fact would
need an adapter (`price_mean` × `qty`) for no gain. The fact table remains the
analytics-facing model and is what week 7's elasticity work uses.

**Deviation on `n_articles`.** `daily_customer_agg` counts distinct articles
*that day*; the rolling layer sums those, so `cust_n_articles_30d` counts
(day, article) pairs rather than distinct articles over 30 days. An exact
distinct count over a range frame is not expressible as a window aggregate. The
name is left visible rather than hidden behind `n_distinct_articles`.

**`avg_price` over a window is computed from the two sums over the same frame**,
not by averaging the daily averages — the latter weights days equally instead of
transactions. Worth stating because it looks like a rename and is not.

### Checkpoint 2.2 — row counts per grain, plus a hand spot-check

Full 31,788,324-row log:

```
daily_customer_agg   9,080,179 rows   22.1 s
daily_article_agg    7,443,545 rows   10.3 s
daily_cross_agg     19,980,389 rows   25.6 s
```

Spot-check on the heaviest customer in the dataset
(`be1981ab818cf4ef6765b2ecaea7a2cbf14ccd6e8a7ee985513d9e8e53c6d91b`, 1,895
transactions), feature row for `day_index = 375` (2019-09-30):

```
pipeline:  cust_n_txn_7d = 15   cust_spend_7d = 0.5082372881355932
by hand:   count(*)      = 15   sum(price)    = 0.5082372881355932
           over raw rows with day_index in [368, 374]
```

Exact to the last digit, and the window is `[d−7, d−1]`, not `[d−7, d]`. **PASS.**

### Checkpoint 2.3 — GATE 1

```
tests/test_pit.py::test_window_excludes_same_day              PASSED
tests/test_pit.py::test_future_events_do_not_change_past_features PASSED
2 passed in 10.42 s
```

**GATE 1 PASSED.**

A passing test proves nothing about a test's teeth, so both were
**mutation-checked** against the two leaks they exist for:

| Mutation | test 1 | test 2 |
|---|---|---|
| `rangeBetween(-w, -1)` → `rangeBetween(-w, 0)` (the one-character leak) | **FAIL** | pass |
| a global mean over all time added to a feature column | **FAIL** | **FAIL** |
| neither (restored) | pass | pass |

That table is the doc's claim about the two tests, confirmed rather than
repeated: test 1 catches the off-by-one and is blind to the global statistic;
test 2 catches the global statistic. Neither test alone is the gate — the pair
is. Run this mutation check yourself when you re-implement; it takes two minutes
and it is the only evidence that the gate is a gate.

## Step 2.4 — Persist, split, backfill

Split constants live in `src/marketrank/splits.py`, exactly the doc's six slices,
with a `CONSUMED_BY` map next to them so a later week cannot quietly borrow
`ope_env`.

### Full build

```
FULL_BUILD_SECONDS 367.8   (6 min 08 s, 8 cores, local[*])
local.features.feature_customer_daily   9,080,179 rows
local.features.feature_article_daily    7,443,545 rows
local.features.feature_cross_daily     19,980,389 rows
warehouse total after the build: 1.0 GB
```

(An earlier full build with DOUBLE money took 283.8 s. Decimal arithmetic costs
about 30% here — see below for why it is worth it.)

### THE FINDING OF THIS STEP — the backfill checkpoint failed, and not for any reason the doc names

First run of "recompute an arbitrary 30-day window mid-2019 and diff it against
what the full run produced":

```
window 2019-06-01 .. 2019-06-30
rows in window: 469,829 (customer)
assert_identical(before, after, ignore_cols=())
  only in a: 31,387
  only in b: 31,387       -> AssertionError
```

Same row count, so nothing was missing — 31,387 rows had different *values*.
The doc predicts this check catches the truncation trap, an off-by-one, or
non-determinism. It was the third, in a form the doc does not mention. Column by
column:

```
COL cust_n_txn_7d        n_diff=0
COL cust_n_articles_7d   n_diff=0
COL cust_spend_7d        n_diff= 7,328   max|delta| = 8.88e-16
COL cust_avg_price_7d    n_diff= 6,750   max|delta| = 2.78e-17
COL cust_n_txn_30d       n_diff=0
COL cust_spend_30d       n_diff=15,631   max|delta| = 8.88e-16
COL cust_n_txn_90d       n_diff=0
COL cust_spend_90d       n_diff=23,758   max|delta| = 1.78e-15

EXAMPLE  day_index=256  full=1.2807966101694914  backfill=1.2807966101694916
                        n_txn_90d = 44 in both
```

**Every integer count is bit-identical; only the DOUBLE columns differ, in the
last bit.** Floating-point addition is not associative, and the summation order
inside a hash aggregate depends on how the input was partitioned. The full build
reads 734 days and the backfill reads 120, so Spark splits the input differently,
sums the same prices in a different order, and lands on a different last bit.

This matters more than the magnitude suggests. Nothing downstream would notice
1e-15. What *would* happen is that the checkpoint fails on every backfill
forever, gets labelled flaky, and is quietly downgraded to a row-count check —
at which point the truncation trap it exists to catch walks straight through.
The doc says this single diff "catches the truncation trap, the off-by-one, and
any non-determinism, which is why it's the checkpoint". It is right, and the
non-determinism it catches first is one you have to design out before the check
is usable.

**Fix: money is `DECIMAL(12,10)`, not `DOUBLE`, from the daily aggregate up.**
Decimal addition is exact and therefore order-independent. `price` has 9,857
distinct values in [1.69e-05, 0.5915] and all survive `DECIMAL(10,8)` without
collision (step 1.4), so 10 decimal places is comfortably lossless. Spark then
carries `spend` as `decimal(32,10)` and `avg_price` as `decimal(38,16)`.

This is the same argument as step 1.4's `DECIMAL` over `DOUBLE` advice for a key
column, arriving one layer later and for a different reason — there the risk was
a float key re-serializing; here it is a float *aggregate* being non-reproducible
across partitionings. Worth noticing that the doc makes the argument for keys and
not for measures, when measures are where it actually bit.

### Checkpoint — after the decimal fix

```
window 2019-06-01 .. 2019-06-30, backfill via build_features(start, end)
BACKFILL_SECONDS 51.5
rows written in the window: 1,960,023
  feature_customer_daily   469,829 -> only in a: 0, only in b: 0   IDENTICAL
  feature_article_daily    356,254 -> only in a: 0, only in b: 0   IDENTICAL
  feature_cross_daily    1,133,940 -> only in a: 0, only in b: 0   IDENTICAL
totals unchanged: 9,080,179 / 7,443,545 / 19,980,389
```

Whole-row comparison with `ignore_cols=()`, i.e. including nothing excluded.
**PASS.**

### And the checkpoint has teeth — mutation-checked

`read_start = start - MAX_WINDOW` was changed to `read_start = start`, which is
the backfill-truncation trap exactly as the doc describes it:

```
only in a: 326,130
only in b: 326,130      -> AssertionError
```

**326,130 of 469,829 customer-day rows in the window — 69% — come back wrong**,
with no error, no null, and a table that still passes its leakage tests. That is
the number that makes the read/write asymmetry worth writing in the docstring.
(The mutated run corrupts the window, so the correct backfill has to be re-run
afterwards to restore it; it was, and the diff is clean again.)

### Numbers §7's metric table asks for

| Quantity | Value |
|---|---|
| Full feature build, 734 days | **367.8 s** |
| Rows written, full build | **36,504,113** across three tables |
| Source rows processed, full build | 31,788,324 |
| 30-day backfill | **51.5 s** |
| Rows written, 30-day backfill | **1,960,023** |
| Machine | 8 cores, 16 GiB, `local[*]` |

Week 4's single-node-vs-multi-node decision is made from these: 6 minutes for the
whole feature layer on a laptop means the feature backfill is **not** the
cluster-shaped job. Candidate generation still is.

---

## Step 3.1 — Baselines first

Measured on `val_tune` (2020-08-12 .. 2020-08-25), **full scale**, all customers,
**464,982** distinct true (customer, article) purchase pairs. Candidates are
built from `t_dat < 2020-08-12` only. Popularity is the top 2,000 articles by
transaction count in the preceding 30 days; repurchase is the customer's own
prior distinct articles, most-recent-first; the union puts repurchase first and
fills with popularity.

Recall is computed by ranking each true pair directly rather than materialising
300k x 500 candidate rows — same answer, and it runs in 92 s instead of not at
all on this machine.

| Baseline | recall@12 | recall@100 | recall@500 |
|---|---|---|---|
| repurchase | 2.32% | 3.28% | 3.36% |
| recent popularity | 1.22% | 6.21% | 17.93% |
| **union (the denominator)** | **2.52%** | **6.91%** | **18.85%** |

**These are the numbers the two-tower has to beat.** Any "recall@N up x% over
baseline" claim in §11 is relative to `union`, not to `repurchase` and not to
random.

### The doc is wrong about repeat purchase, and it is worth knowing exactly how

Step 3.1 says "Repeat purchase is a large fraction of H&M's signal." Measured, it
depends entirely on what counts as "the same thing":

```
REPURCHASE CEILING (fraction of val_tune purchases the customer had EVER bought before)
  same article_id       3.36%   (15,615 / 464,982)
  same product_code     8.22%   (38,203 / 464,982)
  same product_type_no 64.34%   (299,182 / 464,982)
```

`article_id` in this dataset is a specific colour/size variant, so exact-article
repurchase is **3.4%**, not "a large fraction". At the garment level it is 8%; at
the category level it is 64%. The doc's sentence is true only at the category
grain, which is a different mechanism — it is not repurchase, it is category
affinity, and it is the thing step 4.1's third candidate source ("top popular in
the customer's dominant category") actually exploits.

Two things follow, and both change decisions:

1. `rep_rank@500` = 3.36% is a hard ceiling, not a tuning target. No amount of
   ordering the repurchase list improves it.
2. Step 4.1's candidate sources should be read as: ANN, exact-article repurchase
   (small), and **category-conditioned popularity (the big one)**. Budget effort
   accordingly.

Cold start is *not* the explanation for the low numbers: **93.35%** of the true
pairs belong to customers who already had purchase history, and restricting to
those customers changes union recall@500 from 18.85% to 18.82%. The median such
customer has 30 distinct prior articles (p90 = 107, p99 = 250).

Popularity's own ceiling at depth 2,000 is **41.38%** — that is the fraction of
val_tune purchases that are of an article in the recent top-2,000 at all.

### Platform note that costs an hour if you hit it

The macOS `lightgbm` wheel links `@rpath/libomp.dylib` and searches only Homebrew
and MacPorts prefixes, so `import lightgbm` fails with `Library not loaded:
@rpath/libomp.dylib` unless libomp is installed system-wide. `torch` ships one.
`DYLD_LIBRARY_PATH` is read by dyld at process start, so this cannot be fixed
from `config.py` the way `SPARK_CONF_DIR` is — it has to be exported, and the
Makefile does it. A `ctypes.CDLL` preload of torch's libomp does **not** work;
tried and recorded here so nobody tries it twice.

---

## Clarification on step 2.4's DECIMAL change (it is not a step-1.4 revision)

To remove any ambiguity: the `DOUBLE` -> `DECIMAL(12,10)` change in step 2.4 is
about the **feature pipeline's money measures** (`spend`, `avg_price` in
`features.py`), and it is fully landed in commit `edfb37c` with the backfill
checkpoint re-run and passing.

**Step 1.4's grain decision is unchanged**: `price` is still NOT in the
`fact_transaction` key, and `dbt/models/marts/fact_transaction.sql` is untouched
by this. The `DECIMAL(10,8)` measurement in step 1.4 was a "what if we had kept
price in the key" check, and its answer (no collisions) is recorded there as
context, not as a change. Two different columns, two different reasons, one
shared lesson about floats.

---

## Step 3.2 — The two towers

### CHECKPOINT: **FAILED.** The two-tower does not beat the baseline union.

The doc's checkpoint is "Recall@100 on `val_tune` beats the baseline union by a
margin you'd defend." Measured, on the identical evaluation cohort and the
identical denominator (20,000 sampled `val_tune` customers, **70,715** true
(customer, article) pairs):

| | recall@12 | recall@100 | recall@500 |
|---|---|---|---|
| repurchase | 2.322% | 3.309% | 3.394% |
| recent popularity | 1.222% | 6.250% | 17.978% |
| **baseline union** | **2.511%** | **6.967%** | **18.993%** |
| **two-tower (best epoch)** | 1.209% | **5.531%** | 15.148% |

**The two-tower loses to the union on every cutoff, and loses to popularity
alone at 100 and 500.** It is not close enough to call a tie: recall@100 is
5.531% against 6.967%, i.e. **21% worse**, in the direction the checkpoint was
written to catch.

Per the doc, this result is reportable as-is and it is what the step is for —
"a two-tower model that doesn't beat them is a two-tower model you shouldn't
ship", and finding that out in week 3 is the cheap version. **No number here was
tuned until it won.** What follows is what was tried, in the order it was tried.

### Run configuration — REDUCED SCALE, and here is exactly how

| | Full-scale intent | What actually ran |
|---|---|---|
| Training positives | all `train`-slice purchases (~28M) | **2,900,248** rows |
| Customer cohort | all 1,371,980 | **300,000** sampled by hash |
| Training window | 2018-09-20 .. 2020-08-11 | **2020-02-14 .. 2020-08-11** |
| Eval customers | all `val_tune` buyers | **20,000** sampled by hash |
| Embedding dim | 64 or 128 | **64** |
| Epochs | to convergence | **8**, best at epoch 4 |
| Device | GPU | Apple **MPS** |

Wall clock: export 185.5 s, training 862.2 s for 8 epochs (~90 s/epoch)
including a per-epoch `val_tune` evaluation (~11 s each).

`val_tune` is the slice `splits.py` allocates for early stopping, so selecting
epoch 4 on it is the intended use, not a leak. `val_calib`, `ope_env`, `test`
and `holdout` were not read.

Per-epoch recall@100: 4.85, 5.09, 5.27, 5.40, **5.53**, 5.49, 5.51, 5.48 — it
plateaus by epoch 4 and does not trend up. More epochs are not the missing
ingredient; a 20-epoch run on the smaller 479k-row set was *worse* than a
5-epoch run on it (11.2% vs 12.9% recall@500), which is overfitting.

### The logQ correction — measured, and the effect is enormous

Ablation on the 50,000-customer / 479,587-row set, 20 epochs each, everything
else identical:

| | final loss | recall@12 | recall@100 | recall@500 |
|---|---|---|---|---|
| **with logQ** | 6.2530 | 0.822% | 4.054% | **11.220%** |
| without logQ | 6.2025 | 0.013% | 0.105% | **0.669%** |

**A 17x difference in recall@500, and the run without logQ has the *lower*
training loss.** That pairing is the whole lesson in one table: in-batch
negatives make popular articles appear as negatives in proportion to their
popularity, the model learns to push them down, the softmax loss is perfectly
happy about it, and retrieval — which on this dataset is dominated by popular
articles — collapses. Subtracting `log(sampling_prob(article))` from the logits
is three lines and it is the difference between a model and rubble.

This is a stronger result than the doc's framing ("biased toward popularity")
suggests, and it is worth having the number rather than the adjective.

### Two bugs found, both silent, both worth naming

1. **logQ applied in the wrong units.** First implementation was
   `(u·v − log q) / T`. The correction belongs in logit units, *after*
   temperature scaling: `(u·v)/T − log q`. With `T = 0.05`, dividing a
   `log q ≈ −11` term by T produces a ±230 offset that swamps the dot products
   entirely. Symptom: training loss of **49** and recall@500 of **3.5%**. It
   trains, it converges, and it is nonsense.
2. **Off-by-one between `article_idx` and array position.** `article_idx`
   starts at 1 (0 is the padding slot), so a densely packed catalog array of
   105,542 rows is shifted by one against every index in the data — and, worse,
   `topk()` returns positions that are one less than the `article_idx` the
   ground truth is keyed on. **Recall goes to approximately zero and nothing
   raises.** `load_articles()` now returns arrays indexed *by* `article_idx`
   with row 0 reserved, and `recall_at` masks column 0. If you re-implement
   this, that alignment is the first thing to assert.

Both of these are the same genre as the PIT window bound: one character or one
index, no error, plausible-looking output. The difference is that week 2 had a
test for its version and week 3 did not.

### What I would change next, and why I did not

The doc says "if it doesn't [beat the baseline], the problem is almost always
negatives or the item-average feature, not the architecture". Both remain
plausible and neither was resolved:

- **Negatives.** In-batch negatives at batch 1024 sample ~1000 negatives from
  the *positive* distribution. Mixing in uniformly-sampled negatives from the
  full 105k catalog is the standard next move, and it is a ~10-line change.
- **The item-average.** `recent_articles` is capped at the last 20 distinct
  articles in `[d−90, d−1]`, and the median customer has 30 distinct prior
  articles over their whole history. That cap may be discarding most of the
  signal for exactly the heavy customers who generate most of the purchases.
- **Scale.** 2.9M positives over 6 months is ~10% of the available training
  data. The trend across the 479k-row and 2.9M-row runs was up (12.9% -> 15.1%
  recall@500), so more data plausibly closes some of the 3.8-point gap.

I stopped rather than continue tuning because the brief's rule is that a smaller
amount finished honestly beats coverage bought by weakening a checkpoint, and
because week 4 onward is built on top of this. **Anyone continuing from here
should treat step 3.2 as open, not done.**

## Step 3.3 — The ANN index

Measured on the 105,543 x 64 article matrix produced by the model above (the
index question is independent of whether the vectors are any good — it is about
search, not relevance). CPU for both sides, so the comparison is like with like;
`hnswlib` has no GPU path.

```
vectors: 105,543 x 64 float32 = 27.0 MB
HNSW build (M=32, ef_construction=200): 9.33 s

BATCHED, 2,000 queries at once, k=100          ms/query   index recall vs exact
  exact (one matmul)                             0.147           1.000
  HNSW ef=100                                    0.0247          0.9487
  HNSW ef=200                                    0.0323          0.9862
  HNSW ef=400                                    0.0824          0.9958

SINGLE QUERY, 500 queries one at a time, k=100
  exact           p50 1.259 ms   p95 1.818 ms
  HNSW ef=200     p50 0.163 ms   p95 0.270 ms
```

"Index recall" is agreement with exact search, **not** recommendation recall.
Conflating the two is how an ANN section stops meaning anything.

**The doc is right and the numbers say so plainly: ANN is not needed here for
speed.** Exact inner-product search over the whole catalog is 1.26 ms per
one-off query and 0.147 ms amortized. The defensible sentence, with this build's
measured values:

> At 105k items exact search is 1.26 ms p50 / 1.82 ms p95 per query and HNSW is
> 0.16 ms p50 / 0.27 ms p95 at 98.6% index recall — an 7.7x median speedup on a
> stage that was already sub-2 ms. The index earns its place in the serving
> path's tail latency, not in its median, and at 105k items it does not earn a
> place in the architecture story at all.

Note also that the batched exact number (0.147 ms) is only 6x slower than
batched HNSW, because a 2,000 x 105,543 x 64 matmul is exactly what BLAS is for.
The gap opens up at single-query latency, which is what a serving path actually
sees — worth knowing which of the two numbers you are quoting.

What changes at 10M items: the exact matmul is 95x more work and the vectors are
2.6 GB, so it stops fitting comfortably in cache and the single-query cost goes
to ~100 ms. That is where the index stops being decoration.

---

## Step 4.1 — Candidate sources

Reduced scale, same cohort as week 3 so the numbers are comparable: **20,000**
`val_tune` customers, candidates fixed as of 2020-08-12, **70,715** true
(customer, article) pairs. The doc's three sources, each tagged.

```
n_customers                      20,000
n_candidate_rows              2,091,944
mean candidates per customer      104.6   (target N ~ 100)
min / max per customer             50 / 120
rows from ann / repurchase / category   1,000,000 / 418,242 / 742,573
mean sources per candidate          1.033
```

### RECALL CEILING = 7.475%

This is the hard ceiling on end-to-end recall for this candidate set. The ranker
cannot recover a purchase that stage 1 dropped, so **no amount of week-5 work
can push end-to-end recall above 7.5% on this configuration.**

Contribution of each source to the union (a pair can be covered by more than
one, which is why these sum to more than the ceiling):

| Source | covers | alone |
|---|---|---|
| two-tower ANN (top 50) | 3.329% | 3.329% |
| repurchase (top 30) | 2.916% | 2.916% |
| dominant-category popularity (top 40) | 2.162% | 2.162% |
| **union** | **7.475%** | |

Overlap between sources is small: the three solo numbers sum to 8.407% against a
7.475% union, and mean sources per candidate is 1.033. The sources are close to
disjoint, which is the good case — each is buying something the others are not.

### The source list in the doc is measurably suboptimal, and one experiment fixes it

Step 3.1 measured that the *global* recent-popularity list covers 6.25% of these
pairs at depth 100. The doc's third source is popularity **within the customer's
dominant category**, and at depth 40 that covers only 2.162%. So the
personalisation in that source is costing recall rather than buying it — a
single `product_type_no` is too narrow a cone.

Adding plain global recent popularity (top 40) as a fourth source:

```
mean candidates per customer   104.6  ->  131.4
RECALL CEILING                 7.475% ->  9.083%
global popularity alone (depth 40): 3.168%   vs   dominant-category (depth 40): 2.162%
```

**+1.6 points of ceiling for 27 more candidates.** Global popularity at equal
depth beats dominant-category popularity outright, so the honest reading is that
the doc's source 3 should be *both*, or should be replaced. Recorded as a
measured deviation rather than applied silently — the committed
`candidates.py` still implements the doc's three sources, and the fourth is an
experiment whose numbers are above.

### What the ceiling says about the project

7.5% (or 9.1%) is low, and it is the single most important number week 4
produces. It is consistent with everything measured so far: the union baseline
tops out at 18.99% at N=500, exact-article repurchase has a 3.36% ceiling in
principle, and the two-tower underperforms popularity. The bottleneck in this
build is **stage 1**, not the ranker — which is exactly the diagnostic the doc
says the ceiling exists to give, and it is pointing at step 3.2's failed
checkpoint from a second direction.

---

## Step 4.2 — The join, and the size

### Sizing arithmetic, done before the join, from measured row counts

```
train-slice positives (distinct customer x article x day, <= 2020-08-11)
                                                        27,155,032
N candidates per positive                                      100
candidate rows at full scale                        2,715,503,200
feature width after the join                     37 columns (measured)
  of which numeric feature columns                              27
bytes per row, 8-byte numerics + two 64-char id strings    ~ 350 B
                                                        -----------
uncompressed candidate-feature table              ~ 950 GB
at parquet+zstd, using this build's measured 8.6x on the
feature tables (36.5M rows -> ~430 MB on disk)              ~ 110 GB
```

The spec's estimate is 40–160 GB and the measured arithmetic lands inside it, at
**~110 GB compressed / ~950 GB uncompressed** if all of the `train` slice is used
as positives. That is the number that makes "I used Spark because the join
fan-out is ~110 GB" a sentence with a measured `__` in it — and it is also,
unambiguously, **not runnable on this machine** (8 cores, 16 GiB, 13–21 GiB free
disk). Decision #3 that the doc defers to the reader — how much history becomes
training positives — is exactly the knob that sets this, and this build turns it
all the way down.

**What actually ran: 2,091,944 candidate rows for 20,000 customers on one day.**
That is 0.077% of the full-scale row count. Everything below is measured on it
and is labelled reduced-scale wherever it appears.

### The join

```
candidates in                     2,091,944
joined rows out                   2,091,944     (ROWS_LOST_IN_JOIN = 0)
columns after the join                   37
```

Left joins throughout, so no row is lost; the article dimension is broadcast
(it is ~100 MB against a default `autoBroadcastJoinThreshold` of 10 MB, so this
does **not** happen by itself — `F.broadcast()` is explicit in the code).

### NULL AUDIT — and it fails, for the reason the doc predicted in step 2.3

```
cust_*  features NULL on 1,791,222 of 2,091,944 rows   85.62%
art_*   features NULL on   242,961 of 2,091,944 rows   11.61%
cross_* features NULL on 2,029,939 of 2,091,944 rows   97.04%
```

**This is the missing spine.** `build_features` was run with `spine=None`, so
`feature_customer_daily` contains a row for (customer, day) only where that
customer transacted. Candidates are scored on a day the customer mostly did not
transact, so the join finds nothing. Confirmed exactly:

```
customers in the eval cohort                              20,000
with a customer-feature row on 2020-08-12                  2,818   (14.09%)
```

The doc says: "Get this wrong and features exist only where labels are positive
— which is its own, extremely flattering, kind of leak." It is right, and the
number is 85.6%.

### The leakage re-test on the joined table — and the distinction that matters

Re-running step 2.1's test 2 on the joined table **failed at first**: 300,722
rows differed between features built from the full log and features built from a
log truncated at the feature date. Diagnosed rather than assumed:

```
feature rows at day0 from the FULL log            2,818
feature rows at day0 from the TRUNCATED log           0
rows present in both                                  0
value mismatches among shared rows                    0
missing rows that transacted ON day0              2,818 of 2,818
```

**No feature value changed. The row SET changed.** With `spine=None`, whether a
(customer, day) feature row *exists* depends on whether the customer transacted
that day — which is future information relative to the feature date, even though
every value in the row is strictly prior. So:

- The PIT *values* survive the join. Gate 1 is not in question.
- The PIT *row set* does not, and that is a second, distinct property nobody
  states. An explicit spine makes the row set a function of the query rather
  than of the outcome, and that is a better reason to have one than "otherwise
  some rows are missing".

Worth noting why `tests/test_pit.py::test_future_events_do_not_change_past_features`
does not catch this: it compares rows with `day_index <= cutoff`, where both runs
have the same row set. The test is correct about values and silent about
existence. **If you re-implement this, add a third assertion — that the row set
for days <= cutoff is also unchanged — and expect it to fail until the spine
exists.**

### The fix, plumbed and proven at reduced scale

`build_features` now takes `customer_spine` / `article_spine`, and
`features.customer_day_spine(spark, customers, start, end)` builds the pairs.
Proven on the eval cohort at day0:

```
coverage of the 20,000-customer cohort   14.09%  ->  100.00%
value mismatches on the 2,818 rows present both ways:      0
spine rows with non-zero 90-day activity:   14,875 (74.38%)
```

The spine adds rows and changes no value, which is exactly the required
behaviour. The remaining 25.6% of spine rows are customers with genuinely no
activity in the prior 90 days, correctly zero-filled rather than null.

**Not run: a full rebuild of the feature tables with a spine.** The three tables
took 6 min 08 s to build without one; a spine over all 1.37M customers x 734 days
is 1.006 **billion** (customer, day) rows before any features are attached, which
is not buildable here — the practical version is a spine over the candidate
cohort and the scoring days only, which is what the reduced run above does. That
scoping decision is unavoidable at any scale, not an artifact of this laptop: a
dense customer x day spine is never the right object.

---

## Where this build stops, and what is left

Committed and checkpointed: **week 1 complete, week 2 complete (gate 1 passed),
week 3 measured with step 3.2's checkpoint failing, week 4 through step 4.2's
audits.**

| Step | State |
|---|---|
| 1.0 – 1.7 | done, checkpoints passed; 1.7 verified locally, never on a runner |
| 2.0 – 2.4 | done, checkpoints passed; **gate 1 passed and mutation-checked** |
| 3.1 | done, baselines measured at full scale |
| 3.2 | **checkpoint FAILED** — two-tower loses to the baseline union |
| 3.3 | done, ANN measured; conclusion is that ANN is not needed here |
| 4.1 | done, recall ceiling 7.475% measured |
| 4.2 | sizing done, join done, **null audit fails on the missing spine**; fix plumbed and proven at reduced scale, full rebuild not run |
| 4.3 (misha) | **skipped by instruction** — single machine only |
| 5.x – 10.x | **not started** |

**Gate status.** Gate 1 (PIT leakage, step 2.3) — **passed**, and the tests were
mutation-checked so the pass means something. Gate 2 (calibration plot, step 5.3)
— not reached. Gate 3 (OPE confidence bands, week 8) — not reached.

**The honest summary of the modelling half:** every retrieval number this build
produced points the same way. Exact-article repurchase has a 3.36% ceiling, the
two-tower loses to recent popularity, and the three-source candidate set tops out
at a 7.5% recall ceiling. Stage 1 is the bottleneck, and stage 2 cannot be
evaluated meaningfully until it is fixed — a ranker trained on a candidate set
with a 7.5% ceiling is being asked to order a set that usually contains nothing
worth ordering. **Do not read week 5 onward as blocked on effort; it is blocked
on step 3.2's checkpoint, which is the doc's own stated stopping rule.**

Three concrete things the next session should do, in order:

1. **Rebuild the feature tables with a spine over the candidate cohort**, and add
   the row-set assertion to `test_pit.py`. Everything downstream reads NULLs
   until this lands.
2. **Fix stage 1 before building stage 2.** In priority order: mixed negatives
   (in-batch plus uniform from the full catalog), raise or remove the 20-article
   cap on the customer tower's item average, and train on more than 10% of the
   available positives. Add global popularity as a candidate source — measured,
   +1.6 points of ceiling for 27 more candidates.
3. **Only then week 5.** The calibration gate is meaningful; NDCG on a 7.5%-ceiling
   candidate set is not.

### Things that cost real time here, so you can skip the cost

- `spark-defaults.conf` cannot carry an env-dependent warehouse path (step 1.5).
- dbt-spark's session needs `SPARK_LOCAL_IP` because it never calls `get_spark()`.
- Iceberg has no views; staging must not be `view` (step 1.5).
- Money must be DECIMAL or the backfill checkpoint can never pass (step 2.4).
- macOS lightgbm needs `DYLD_LIBRARY_PATH` pointing at torch's libomp (step 3.1).
- logQ belongs after temperature scaling, and `article_idx` is 1-based (step 3.2).
- The spine is not optional the moment week 4 starts (step 4.2).

---
---

# Stage-1 recovery — `docs/STAGE1_RECOVERY.md`, steps R.0–R.6

Everything above this line is the reference build, and the "Where this build
stops" table is its state at the moment the recovery plan was written. The plan
inserts between week 3 and week 5 and is executed below, in its order: R.0
(tests) before any compute, R.4 (scale) last.

Same rules as above. Every number was produced by a command that ran; anything
not produced says **not run**. Reduced-scale configurations state exactly what
was reduced, and every number derived from one carries the label.

## Environment for the recovery session

| Fact | Value |
|---|---|
| Machine | as above — Apple Silicon, 8 cores, 16 GiB RAM |
| Free disk at R.0 | **22 GiB** on the volume holding the worktree |
| Warehouse size at R.0 | **2.8 GB** |
| `MARKETRANK_DATA_RAW` | `/Users/test/Developer/marketrank/data/raw` (read-only) |
| `MARKETRANK_WAREHOUSE` | `/Users/test/Developer/marketrank-build/warehouse` |

Disk is the binding constraint on R.1's rebuild and it is watched at every step:
the standing instruction is to stop at a clean committed step if free space drops
below 10 GiB.

### The entry number, reproduced before anything was touched

`artifacts/twotower/model.pt` was re-evaluated on the untouched features, the
untouched 20,000-customer cohort and the untouched 70,715-pair denominator:

```
REEVAL_OLD_FEATURES  n_true_pairs 70715
  recall@12 1.1327%   recall@100 5.4812%   recall@500 15.0039%
```

**That is epoch 7, not epoch 4.** Compare `artifacts/twotower/history.json`:
epoch 7 is `recall_at_100 = 0.054811567559923634` to the last digit, and epoch 4
— the **5.531%** headline the plan quotes — is `0.055306512055433785`.

So the saved checkpoint is the *last*-epoch model and the best-epoch model was
never written to disk. `train()` returns the model after the final epoch and
nothing in the code path saves per-epoch. This matters for R.1, which says
"re-run 3.2's evaluation bit-for-bit unchanged, same checkpointed model": the
only checkpoint that exists is epoch 7's, so the re-eval arm is compared against
**5.481%**, and the retrain arm — which selects its own best epoch — is compared
against **5.531%**. Both reference numbers are carried explicitly below rather
than collapsed into one, because collapsing them would silently move the bar by
0.05 points in whichever direction flattered the result.

Two things this also establishes: the eval path is deterministic (it reproduced
a four-week-old number to 16 significant figures), and the artifacts on disk are
intact despite `artifacts/` being gitignored.

## Step R.0 — Tests before compute

**Written and committed before any recovery compute ran**, which is the point of
the step: week 3 shipped two silent bugs and had no tests where week 2 had its
leakage pair, so every R.1–R.4 number is worthless if a third one is still in
the eval path.

`tests/test_retrieval.py`, four assertions. Three run with no JVM and no data;
the coverage test builds its own four-row DataFrame, so the whole file runs in
CI alongside `test_pit.py`.

### Two small pieces of production code moved so they could be tested

1. **`retrieval/model.py::sampled_softmax_logits`** — the logit construction was
   inline in `train()`, where no test can reach it. Extracted verbatim; the
   placement of the logQ term relative to the temperature is now an executable
   claim instead of a comment. R.3 extends this same function with uniform
   negatives, so the extraction pays for itself twice.
2. **`features.py::feature_coverage`** — the coverage audit, phrased over the
   feature table. Test 2 and R.1's full-scale checkpoint call the identical
   function, which is the only way the test can be said to be the audit.

### Why the spine bug was invisible to week 3 in the first place

Worth stating because it changes where the audit has to live.
`retrieval/dataset.customer_context` left-joins the customer features and then
wraps every numeric in `log1p(coalesce(col, 0.0))`. A customer with no feature
row therefore reaches the tower as a **row of plausible-looking zeros, not as a
null** — there was never a null downstream to audit. Week 4 only caught it
because the candidate join does not coalesce. So the audit has to be taken
against the feature table, before anything fills the gaps in, and that is what
`feature_coverage` does.

### Checkpoint — the mutation table

A passing test proves nothing until it has failed for the right reason, so both
known bugs were re-introduced, one line each, plus the spine and the padding
mask. Restored between every run; `grep -rn MUTANT src/` is empty afterwards.

| Mutation | t1 align | t2 coverage | t3 recall | t4 logQ |
|---|---|---|---|---|
| *(none — restored)* | pass | pass | pass | pass |
| `load_articles` returns densely-packed arrays | **FAIL** | pass | pass | pass |
| `recall_at` stops masking column 0 | **FAIL** | pass | pass | pass |
| logQ correction divided by the temperature | pass | pass | pass | **FAIL** |
| `rolling_features` ignores the `spine` argument | pass | **FAIL** | pass | pass |

Every mutation is caught, each by exactly one test, and no test is redundant
with another. **PASS.**

Full suite after restoration: `pytest -q` → **9 passed in 18.52 s** (the 5
pre-existing tests plus these 4).

### DEVIATION — the plan's checkpoint cannot hold as written, so a fourth test exists

The plan says: *"Reverting the logQ-units fix fails test 3; reverting the
alignment fix fails test 1."* The second half is true and is confirmed above.
**The first half is not achievable by test 3 as the plan specifies it.**

Test 3 is "a hand-built 3-customer case where the correct recall@k is computable
on paper". Recall is a pure function of fixed embeddings and ground truth; it
never constructs a training logit. The logQ-units bug lives in `train()` and
cannot reach it. The only way test 3 could catch it is if test 3 *trained* a
model — which would make it an assertion about optimisation, flaky and slow, and
would stop it being computable on paper, which was the property the plan asked
for.

Resolved in favour of the plan's intent (both known bugs covered by the suite)
rather than its letter: tests 1–3 are written exactly as specified, and
**test 4, `test_logq_correction_is_applied_in_logit_units`, is added** to carry
the logQ half. It asserts the correction equals `(u·v)/T − log q`, that it does
*not* equal `(u·v − log q)/T`, and that the gap between them is
`log q · (1 − 1/T)` — a **+219.77**-logit offset at `T = 0.05`, which is the
measured mechanism behind week 3's loss of 49 and recall@500 of 3.5%.

Test 3 is kept and is not redundant: it pins the **denominator** to true
(customer, article) pairs rather than customers, which is the property that
makes the tower's numbers comparable to `baselines.py`'s at all. It catches
neither known bug, and the table above says so honestly rather than implying
four tests catch four bugs.

The plan document is not edited — this is recorded here, per the standing rule.

## Step R.1 — Kill the confound: spine rebuild, then the identical eval

**Full scale, and here is exactly what "full scale" means.** The spine covers
**all 1,371,980 customers and all 105,542 articles** on the scoring day, unioned
with the natural transaction-day rows. It is not a dense customer × calendar
spine, and that is a modelling decision rather than a laptop concession: a dense
spine over the whole log is 1,371,980 × 734 = **1.006 billion** (customer, day)
rows before a single feature is attached, and it is never the right object at any
scale. Features are needed on the days something is *scored*, and that set is
small. The cost is linear — ~1.37M customer rows and ~105k article rows per
scoring day — so extending this to the whole 14-day `val_tune` window is a
decision about disk, not about correctness.

Scoring day: **2020-08-12** (`day_index` 692), `val_tune`'s first day, which is
the day every candidate and every recall number in this build is measured on.

```
REBUILD_SECONDS 318.8      (full rebuild, 734 source days, 8 cores, local[*])
feature_customer_daily   9,080,179 -> 10,434,260   (+1,354,081)
feature_article_daily    7,443,545 ->  7,537,467   (+   93,922)
feature_cross_daily     19,980,389 -> 19,980,389   (unchanged — no cross spine)
```

The deltas are exactly right and worth checking rather than trusting:
1,371,980 − 1,354,081 = **17,899** customers already had a row on day 692 (they
transacted that day), and 105,542 − 93,922 = **11,620** articles likewise.

### Checkpoint part 1 — coverage, via R.0's test-2 function at full scale

```
COVERAGE feature_customer_daily  1,371,980 / 1,371,980   100.00%
COVERAGE feature_article_daily     105,542 /   105,542   100.00%
EVAL_COHORT_COVERAGE (the 20,000)   20,000 /    20,000   100.00%
```

**14.09% → 100.00%** on the eval cohort. The same `features.feature_coverage`
call R.0's test 2 makes, so the test really is the audit.

### Checkpoint part 2 — the spine adds rows and changes no value, proven

Not asserted from the reduced-scale proof — measured at full scale against the
**pre-rebuild Iceberg snapshot**, which is what `checks.read_snapshot` is for:

```
feature_customer_daily  (snapshot 7728130163272976666)
  only in BEFORE (a changed or lost value):          0
  only in AFTER  (rows the spine added):     1,354,081
feature_article_daily   (snapshot 5022465635284524252)
  only in BEFORE:                                    0
  only in AFTER:                                93,922
```

Whole-row comparison over 9.08M and 7.44M rows. Zero rows changed. This is the
property that makes the rebuild a *measurement* rather than an improvement.

### THE FINDING OF THIS STEP — the confound was eval-time only, and that was checkable in advance

The training positives sit on days the customer **did** transact, so
`daily_customer_agg` always produced a row for them and the window layer always
emitted one. Measured on the pre-rebuild snapshot, over the reduced run's own
training window:

```
TRAIN_SIDE_COVERAGE_BEFORE_REBUILD  2,158,250 / 2,158,250 = 100.0000%
```

**The training set was never affected by the spine bug.** Only the eval side
was, because that is the only place the model is scored on a day nothing was
bought. So R.1's two arms are not symmetric, and the plan's framing —
"re-eval isolates eval-time contamination; the retrain isolates training-time
contamination" — resolves to: there was no training-time contamination to
isolate, and this number is why.

What the eval cohort's inputs actually looked like, dirty vs clean:

```
cohort identical: True   (same 20,000 customer_ids, same order)
eval rows with ALL-ZERO rolling features:  17,744 (88.72%)  ->  5,125 (25.62%)
eval rows whose numeric features changed:  12,619 (63.09%)
```

The residual 25.62% is not a bug: those are customers with genuinely no activity
in the prior 90 days, now correctly zero-*filled* rather than absent. Note the
dirty figure is 88.72% all-zero against the 85.91% *missing* reported in step
4.2 — the gap is customers who had a row whose windows were all zero anyway.

### The re-export, and the proof that the cohort did not move

```
EXPORT_SECONDS 156.0
EXPORT n_articles            105,542
EXPORT n_train_rows        2,900,248
EXPORT n_eval_customers       20,000
EXPORT n_eval_truth_pairs     70,715
```

**2,900,248 and 70,715 are the reference build's numbers to the row.** The
cohort and the denominator are hash-derived and feature-independent, so this is
the confirmation that every comparison below is on identical ground rather than
an assertion that it is.

### Checkpoint part 3 — the two arms, measured

Both arms on the identical cohort (20,000 customers / 70,715 pairs), exact
top-N over the full 105,542-article catalog.

| | r@12 | r@100 | r@500 |
|---|---|---|---|
| wk3 model (epoch 7), **dirty** features — what was on disk | 1.1327% | 5.4812% | 15.0039% |
| wk3 headline (epoch 4, never saved), dirty | 1.2091% | **5.5307%** | 15.1481% |
| **arm A** — wk3 epoch-7 model, **clean** features | 1.2105% | **5.6438%** | 15.5087% |
| **arm B** — clean retrain, best epoch 5 of 8 | 1.2501% | **5.7852%** | 15.6077% |
| baseline union (the bar) | 2.511% | **6.967%** | 18.993% |

`REBUILD_SECONDS 318.8`, `EXPORT_SECONDS 156.0`, `TRAIN_SECONDS 793.2`.

Arm A was re-measured from the checkpoint rather than carried over: 5.643781%,
which reproduces the first arm-A run to the digit. Arm B's per-epoch recall@100
is `4.9523, 5.2337, 5.5123, 5.6820, 5.7074, 5.7852, 5.6961, 5.7032` against
week 3's `4.8519, 5.0866, 5.2719, 5.4048, 5.5307, 5.4939, 5.5066, 5.4812` —
higher at every epoch, and still plateauing by epoch 5.

### What R.1 actually bought, stated conservatively

**The spine fix is worth about +0.3 points of recall@100 and the tower still
loses to the baseline union by ~1.2 points.** Against the headline the plan
quotes (5.5307%), arm B is +0.25; against what was actually on disk (5.4812%),
+0.30. The gap to 6.967% closes from 1.486 to 1.182 — roughly **20% of the
shortfall**, on the most generous reading.

So hypothesis 1 is **substantially demoted, not eliminated**: the confound was
real, measurable, and small. Whatever explains the remaining 1.18 points is not
the spine. R.2 and R.3 now carry the argument, which is what R.1 existed to
establish.

### THE METHODOLOGICAL PROBLEM R.1 EXPOSED — read this before trusting any later delta

Arm A and arm B disagree at the same epoch index. Arm A is the week-3 epoch-7
model on clean features: **5.6438%**. Arm B's own epoch 7, also clean:
**5.7032%**. If the spine changed nothing on the training side — and
`TRAIN_SIDE_COVERAGE_BEFORE_REBUILD` says it changed nothing, 100% coverage
before the rebuild, zero values altered — then those two models should be the
same model and those two numbers should be equal. They differ by **0.059
points**, and the epoch-7 training losses differ too: 6.5935 (wk3) against
6.5957 (arm B).

The train export reproduced 2,900,248 rows exactly and `seed=0` in both, so the
divergence is not the data. Two candidates remain: week 3's run used a slightly
different configuration, or training is not bit-reproducible on this device
(MPS reductions are order-sensitive). **The first is untestable from the
artifacts** — week 3's `history.json` is a bare list of per-epoch metrics with no
argument vector recorded, which is precisely the gap `train_towers.py` closes by
writing `args` into every `metrics.json`.

Either way the consequence is the same and it governs the rest of the ladder:

> **There is a non-zero run-to-run variance of unknown magnitude, and it is at
> least ~0.06 points of recall@100. Until it is measured, no R.2/R.3 delta
> smaller than it can be attributed to the change that was made.**

R.2's article-volume features and R.3's uniform negatives are both expected to
move recall by more than that — but "expected" is not "measured", and an
ablation ladder whose rungs are inside the noise is exactly the kind of
flattering artifact this project keeps refusing to produce. So a step is
inserted before R.2, as a deviation from the plan:

**R.1b — measure the noise floor.** Re-run the arm-B configuration unchanged at
seeds 1 and 2. The spread across seeds 0/1/2 is the resolution of every
comparison that follows, and it gets quoted next to every later number rather
than assumed away.
