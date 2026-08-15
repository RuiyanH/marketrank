# Implementation guide — `marketrank` (Project D v3)

*Step-by-step build order for [`Project_D_v3_Marketplace_ML_System.md`](../../brainstorm_project/Project_D_v3_Marketplace_ML_System.md). Written 2026-08-14 against the repo as it actually stands, not against week 0.*

**How to read this.** Every step has the same five parts:

| Part | What it's for |
|---|---|
| **Why now** | Where it sits in the pipeline and what breaks if you do it later |
| **Write** | Which file, which functions, what they take and return — a specification, not a finished file |
| **Must know** | The idea you have to be able to defend in an interview. Go slow here |
| **Plumbing** | Copy it, move on. Knowing it cold buys you nothing |
| **Checkpoint** | The observable fact that says the step is done. If it doesn't match, stop |

Some steps open with **Think first** — a question to answer before you read the rest of the step. They're the places where the design decision is more interesting than the code.

**Depth taper, stated honestly.** Weeks 1–5 are written at full step depth because they're next and because gate 1 is where a silent bug poisons everything downstream. Weeks 6–10 are written as concrete steps with the traps named, but thinner — they'll get their own pass when you reach them, and by then the earlier weeks will have changed decisions this document can't predict. Anything marked **VERIFY** is a claim I did not run; check it at the checkpoint rather than trusting it.

---

## 0. Where you actually are

Audited 2026-08-14 from the repo, not from the timeline.

**Done and working:**

- `local.raw.transactions` — Iceberg, Hadoop catalog at `warehouse/`, partitioned by `days(t_dat)`, all **734 days** loaded (2018-09-20 → 2020-09-22).
- `get_spark()` with the Iceberg runtime jar, loopback bind workaround, UTC session timezone.
- `load_transactions()` idempotent via `writeTo(...).overwritePartitions()`.
- `checks.assert_identical()` — snapshot-vs-snapshot row diff. This is your idempotency proof and it's a genuinely good artifact. It needs one refactor (step 1.3) and a pytest wrapper (step 1.6).
- Schema evolution exercised once (`ADD COLUMN promo_flag`).

**Not started:** `raw.articles`, `raw.customers`, the whole dbt project, dbt tests, CI, and everything from week 2 on.

**Four things to fix before moving on** — all small, all real:

1. **The table and the DDL disagree.** `ingest.create_tables()` doesn't declare `promo_flag`, but the live table has it. Rebuild the warehouse from scratch today and you get a different schema than the one you have. Either drop the column or put it in the DDL. Then decide which one is the honest artifact — see step 1.3.
2. **`overwritePartitions()` is idempotent but not late-arrival-safe.** See step 1.2's **Think first**. Not a bug now; it's a week-9 landmine, and knowing it *now* changes what you build in step 1.4.
3. **No ingestion timestamp.** dbt's freshness tests and every week-9 late-arrival check need to know *when a row landed*, which is not the same as when the event happened. Adding it later means rewriting 734 partitions.
4. **`checks.py` and `SETUP_MISHA.md` describe different functions.** `checks.py:22` is `assert_identical(spark, table, snap_a, snap_b)`, but the runbook's Phase 5b calls `assert_identical(a, b)` on two DataFrames and `read_snapshot(spark, table)` with no snapshot id — a refactor that hasn't happened. **This document owns the target signature** (step 1.3), and the runbook has been patched to match it.

---

## 1. Week 1 (remaining) — lakehouse + dimensional model

Target end state: three raw Iceberg tables, a dbt project that builds `fact_transaction` / `dim_customer` / `dim_article` on Spark and the same project on DuckDB in CI, tests green on every PR.

### Step 1.0 — Declare your dependencies

**Why now.** `pyproject.toml` has no `dependencies` block, so `pip install -e .` installs nothing and `pyspark==3.5.9` is pinned only in the prose of `SETUP_MISHA.md` — its Appendix D already flags this. Step 1.7's CI job cannot pass until it's fixed, and neither can a clean rebuild on misha in week 4.

**Write.** `[project] dependencies = ["pyspark==3.5.9"]`, plus a `[project.optional-dependencies] dev` extra holding `dbt-spark[session]`, `dbt-duckdb`, `pytest`, and (later) `lightgbm`, `torch`, `hnswlib`. CI and the laptop install `.[dev]`; misha installs `.`.

**Must know.** Why `pyspark` is a hard dependency and not a dev extra: `get_spark()` is library code, not tooling. If importing `marketrank` requires Spark, the dependency is declared, not documented. Pinning the exact version matters more than usual here because the Iceberg runtime jar coordinate in `config.py` is built against Spark 3.5 / Scala 2.12 — a floating `pyspark` silently breaks that pairing, which is the failure mode `SETUP_MISHA.md`'s Appendix B lists as `NoSuchMethodError`.

**Checkpoint.** A fresh venv plus `pip install -e ".[dev]"` gives you working `pytest` and `dbt` with no other commands.

### Step 1.1 — Load `articles` and `customers`

**Why now.** The two dimension models have no source without them, and `articles` carries the metadata the article tower (week 3) and the ranker (week 5) both live on.

**Write.** In `ingest.py`, alongside the transactions pair:

- `ARTICLES_SCHEMA`, `CUSTOMERS_SCHEMA` — declared explicitly, same as you did for transactions. Never infer.
- `load_articles(spark)`, `load_customers(spark)` — read CSV, write to `local.raw.articles` / `local.raw.customers`, unpartitioned, `createOrReplace()`.

**Must know.** Why these two get `createOrReplace()` and transactions gets partition overwrite: these are *snapshots of current state* (105k articles, 1.37M customers — a few hundred MB), and there is no meaningful "partition" of a dimension. Transactions are an append-structured event log where you want to be able to rewrite one day without touching the other 733. The write strategy follows the grain, not preference. That sentence is an interview answer.

Two schema traps in this data, both of which bite if you let Spark infer:

- `article_id` is a **zero-padded string** (`0663713001`). Inferred as a long, the leading zero is gone and it won't join to `articles.csv` later. You already declared it as `StringType` for transactions — do the same here.
- `customers.csv` has `FN` and `Active` as sparse 1.0/null floats, and `age` has nulls. Declare them nullable, cast at the staging layer, not at read.

**Plumbing.** The CSV read options.

**Checkpoint.** `spark.table("local.raw.articles").count()` = **105,542** and `local.raw.customers` = **1,371,980** — measured 2026-08-14, so these are yours now, not the widely-cited figures. They're what §2's "105k articles / 1.4M customers" claim rests on.

**And a number worth knowing the moment the tables exist:** only **104,547** articles and **1,362,281** customers ever appear in a transaction. So **995 articles never sold** and **9,699 customers never bought anything**. Two consequences you'd otherwise meet as bugs:

- Your `relationships` tests only work in one direction. `fact → dim` passes; `dim → fact` would fail on those rows, and it *should* — a catalog article that never sold is normal, not a referential-integrity violation.
- That's your genuine cold-start population for week 3. A recommender that can only score articles it has seen sell is a recommender that can never surface a new product, which is the failure mode the article-metadata tower exists to avoid.

### Step 1.2 — Decide the write strategy, deliberately

**Think first.** You load day `2019-06-01` from the CSV. Then a corrections file arrives holding three extra transactions for that same day. You run `load_transactions()` on just that file. What is in the table afterward?

<details>
<summary>Answer</summary>

Three rows. `overwritePartitions()` is a *dynamic partition overwrite*: every partition present in the incoming DataFrame is replaced wholesale by the incoming data. The other ~40,000 rows for that day are gone.

This is not a bug in what you built — it's correct for "re-run the full day from the source of truth," which is exactly how you've been using it. It's the wrong shape for "apply a delta." Week 9 is specifically about the delta case, which is why §5's semantics list names `MERGE` as the mechanism.
</details>

**Must know.** The distinction is *full-refresh-per-partition* vs. *merge-on-key*, and the reason it matters is that merge needs a key and your transaction rows **do not have one**. There is no line-item id, and `(customer_id, article_id, t_dat, price, sales_channel_id)` is genuinely duplicated in the source — a customer buying two of the same item on the same day produces two identical rows. This is the central modeling problem of the dataset and step 1.4 is where you resolve it.

**Do now:** nothing to the code. Add a two-line comment on `load_transactions` saying it is a full-day refresh, not a merge, and that late-arrival handling is week 9. Write the constraint down where you'll trip over it.

### Step 1.3 — Add `_ingested_at`, using the schema-evolution mechanism for real

**Why now.** Week 9 needs it; adding it in week 9 costs a full re-load. And it turns the `promo_flag` toy exercise into an evolution you did for a reason.

**Write.**

1. `ALTER TABLE local.raw.transactions ADD COLUMN _ingested_at TIMESTAMP` — and add it to the `CREATE TABLE` DDL in the same commit so table and code agree.
2. In `load_transactions`, `.withColumn("_ingested_at", F.current_timestamp())` before the write.
3. Drop `promo_flag`, or keep it and declare it. Not both states at once.
4. Refactor `checks.assert_identical` to `assert_identical(a: DataFrame, b: DataFrame, ignore_cols=("_ingested_at",))` — see **Must know**, below. `read_snapshot(spark, table, snapshot_id)` keeps its signature.

**Must know.** Two things.

*Why the existing 734 partitions don't need rewriting.* Iceberg schema evolution is a metadata operation — the column gets an id in the new schema, and old data files that lack it read back as null. Nothing is rewritten. That is the whole point of "schema evolution doesn't break downstream readers," and you should be able to say **why** it's free: Iceberg tracks columns by assigned id, not by position in the file, so adding, renaming, and reordering are all metadata-only. Hive-style tables resolve by position, which is why the same operation there is a rewrite or a corruption.

*The write-path consequence.* Once the table has a column the DataFrame doesn't, `writeTo(...).overwritePartitions()` has to decide what to do. **VERIFY this yourself** — write one day from a DataFrame missing `_ingested_at` and see whether Iceberg fills null or refuses. Whichever it does, that behavior is the answer to "what happens when the producer and the table drift apart," and it's a better story if you found out by running it.

**Must know — you just broke your own idempotency proof, and the fix is a definition.** `assert_identical` compares whole rows. `_ingested_at` is `current_timestamp()`, so two identical re-runs now produce two tables that differ in every row, and the check fails forever.

The fix is not to drop the column — it's to notice that *idempotency was never a claim about bytes*. It's a claim about **business content**: re-running a load must not change what the table says about the world. `_ingested_at` says something about the *pipeline*, not the world. So the comparison excludes it, and excludes anything else you later add with a leading underscore.

That distinction — business columns vs. pipeline metadata, with idempotency defined over the former — is worth being able to state cleanly, because the alternative ("my loads are idempotent as long as I don't record when they ran") isn't a property anyone would ship. Make the exclusion a named parameter with a default rather than a hardcoded column name, so the rule reads as a rule.

**Checkpoint.** Old partitions return null `_ingested_at`; a re-loaded day returns a timestamp; and `assert_identical(a, b)` on two consecutive full re-runs **passes** while a whole-row diff of the same two snapshots fails. Both facts, together — the second one is what proves the exclusion is doing work rather than hiding a real difference.

### Step 1.4 — Resolve the grain, then define the fact table's key

**Think first.** What is one row of `fact_transaction`? You have three options and they are not equivalent:

- **(a) One row per source row.** Grain = a line item. No unique key exists, so the dbt `unique` test has nothing to test.
- **(b) One row per (customer, article, day, channel), with `qty` and `price`.** Grain = a basket line. Key exists and is natural. You lose nothing — the duplicate rows carry no information beyond their count.
- **(c) One row per source row with a synthetic surrogate key.**

Which one, and why is (c) a trap here?

<details>
<summary>Answer</summary>

**(b).** And (c) is a trap because any surrogate key you can generate — `row_number()`, `monotonically_increasing_id()`, a hash including a de-dup index — depends on **partitioning and read order**, which Spark does not guarantee across runs. A key that changes when you re-run the job destroys idempotency: re-loading a day produces "the same data" with different keys, so downstream incremental merges see every row as new. You'd have built a key that breaks the one property you spent week 1 establishing.

(b) gives you a key that is a function of the data alone. That is the property you want, and "the key is derived from the business grain, not from row order" is the sentence that says it.
</details>

**Must know — the `price` subtlety, and the answer, measured 2026-08-14.** The same customer buying the same article twice on the same day can pay **two different prices** (a markdown mid-day, or two variants sharing an `article_id`). So does price belong *in* the key?

The measurements, over all 31,788,324 rows:

| | |
|---|---|
| Distinct (customer, article, day) | 28,575,395 |
| …with more than one row (multi-quantity) | **9.51%** — common |
| **…with more than one distinct price** | **0.80%** (1.72% of source rows) |
| …of those, *not* explained by sales channel | **97.9%** |
| Price spread when it happens | median **3.2%**, p90 **25%**, max 97% |

So it's genuine — the channel is already in the key and doesn't account for it — but uncommon, and it mixes two different phenomena. The small gaps are rounding (`0.028457627…` vs `0.028474576…` is 0.06% apart, one cent under a fixed scale factor); the large ones are real markdowns (`0.010661…` vs `0.011847…` is 11%).

**Decision: keep price OUT of the key.** The argument is one line of arithmetic, and it's the one to give in an interview:

> `avg_price × qty` = the sum of the individual prices, exactly. So averaging loses **no revenue**.

Store `qty = COUNT(*)` and `revenue = SUM(price)`, and the money is preserved to the last unit. The only thing you give up is knowing that two units went out at $10 and $8 rather than $9 and $9 — and nothing downstream consumes that. Week 7's elasticity model works at **(article, week)** grain, where those rows are averaged across all customers regardless of what the fact table did.

Carry `price_min` and `price_max` as measures alongside `price_avg` and you haven't even lost the spread — it's just not in the key, which is where it didn't belong. Note the choice in the README anyway; it's exactly the kind of thing an interviewer asks about a fact table.

**The type question closes with it.** A `DOUBLE` is a poor key column — deterministic for one CSV parsed one way, but fragile the moment anything re-serializes it (a Parquet round-trip through a different writer, a `float32` cast, a value that reads back as `0.050830508474576265`), at which point merge and uniqueness silently see two keys where there's one. That would have forced `DECIMAL(10,8)` or the raw string. Since price is now a measure rather than a key, `DOUBLE` is fine.

`fact_transaction` therefore carries: the (b) grain, `qty`, `revenue = SUM(price)`, `price_avg`, `price_min`, `price_max`, and `MAX(_ingested_at)`. Note `revenue` is the **sum**, not `qty × price` — with a mean price they're equal, but writing it as the sum is what makes that true by construction rather than by coincidence.

**Checkpoint.** Your fact table has **28,583,889 rows** against 31,788,324 source rows — a **10.1% collapse**, which *is* the multi-quantity purchase rate. The (b) grain's uniqueness test passes. (For reference: putting price in the key would have given 28,813,419 rows — 0.8% more, for no revenue accuracy.)

**One outlier the count turned up, worth carrying to week 9:** the largest single (customer, article, day) group has **570 rows** — one customer, one item, one day. A bulk order or a test account, either way not organic. That's a ready-made case for step 9.2's row-count anomaly tests, and a better one than a synthetic threshold because it's real.

### Step 1.5 — Stand up dbt on Spark

**Why now.** This is the SQL artifact, and it's the thing that makes the model layer legible to a data interviewer. It's also the fiddliest setup in week 1, so budget for it.

**Write.**

1. Both dbt adapters are already in the `dev` extra from step 1.0 — don't `pip install` them ad hoc.
2. `dbt/` at the repo root: `dbt_project.yml`, `models/staging/`, `models/marts/`, `seeds/`, `profiles.yml` (commit it — no secrets in a local setup).
3. Staging: `stg_transactions`, `stg_articles`, `stg_customers` — rename, cast, no business logic. Views.
4. Marts: `dim_article`, `dim_customer`, `fact_transaction`. Tables (`file_format='iceberg'`).

**Must know — the one real obstacle.** dbt-spark's `session` method builds its **own** SparkSession in dbt's process. `spark.jars.packages` and the Iceberg catalog config must exist *at session construction*, so setting them from `profiles.yml` after the fact does not work. The fix is one config file that both consumers read:

- Create `conf/spark-defaults.conf` with the Iceberg jar coordinate, the extensions class, the `local` catalog config, `spark.sql.defaultCatalog=local`, and the UTC timezone.
- In `config.py`, next to the two that are already there: `os.environ.setdefault("SPARK_CONF_DIR", str(PROJECT_ROOT / "conf"))`.
- Also export `SPARK_CONF_DIR` from the Makefile's `dbt` targets.
- Strip **those** configs out of `get_spark()` so there is one source of truth for them.

**Must know — why both, and why they aren't redundant.** They cover disjoint consumers, and getting this wrong breaks the tools you use most:

- **`config.py`'s `setdefault` covers everything that imports `marketrank`** — a bare `pytest`, the notebook, any REPL that calls `get_spark()`. Without it, the moment you strip the catalog out of the builder, every one of those gets a session with no `local` catalog and fails with `TABLE_OR_VIEW_NOT_FOUND`. An export that lives only in the Makefile reaches only Makefile-launched processes, which is not how you actually run things. The repo already uses exactly this pattern at `config.py:10` for `JAVA_HOME` — same reason (Spark's launcher reads it out of the environment), same mechanism, and import-time is early enough because it runs before session construction. Note it must be `setdefault`, not `os.environ[...] = ...`: a real export from `env.misha.sh` or CI has to win.
- **The Makefile export covers dbt**, which never imports `marketrank` and therefore never runs that line.

With both in place, `SETUP_MISHA.md`'s `env.misha.sh` export becomes belt-and-braces rather than load-bearing — which is the right relationship for a runbook line you'll read once in week 4.

**Must know — but do not strip everything.** "One source of truth" applies to the *catalog*, not to the whole builder, and there are two settings that must stay conditional in `get_spark()`:

- **`spark.driver.bindAddress` / `spark.driver.host` (`spark.py:29`).** These are already guarded by `if master.startswith("local")`, and that guard is load-bearing in both directions. Move them into a static `spark-defaults.conf` and week 4's multi-node standalone cluster breaks — executors on other nodes cannot reach a driver bound to `127.0.0.1`. Delete them and the laptop breaks, because that stale-DHCP hostname is exactly why they exist.
- **`spark.local.dir` (`SETUP_MISHA.md` Phase 3b).** Node-local `/tmp` on misha, a repo-local path on the laptop. Also mode- and machine-dependent.

So the split is by *kind*, not by convenience: **engine-agnostic configuration** (which catalog exists, which jar implements it, what timezone) goes in the conf file, where dbt can see it. **Runtime and topology configuration** (where the driver binds, where spill lands, how much memory, which master) stays in `get_spark()`, where it can branch. That's a defensible line, and it's the answer if someone asks why you didn't just put everything in one file.

**Reconcile the runbook and CI when this lands.** `SETUP_MISHA.md`'s `env.misha.sh` was written before this step existed; it has been patched to export `SPARK_CONF_DIR`, and with the `config.py` default above that export is now redundancy rather than the only thing holding it up. Step 1.7's workflow needs the same treatment — either the `config.py` default carries it (it does, for `pytest`) or the workflow exports it explicitly (needed for any `dbt` step). Check both when you run them.

**VERIFY:** that `spark.sql.defaultCatalog=local` is what makes dbt's unqualified `schema.table` names land in the Iceberg catalog rather than the built-in Hive one. If it doesn't, the fallback is fully-qualifying the catalog in `dbt_project.yml`'s `catalog:`/database config. Find out at the checkpoint, not in week 4.

**Plumbing.** `dbt_project.yml` boilerplate, the `sources.yml` block pointing at `local.raw.*`.

**Checkpoint.** `dbt build` creates `local.marts.fact_transaction` as an Iceberg table — confirmed by `SELECT * FROM local.marts.fact_transaction.snapshots`, which only exists if it's genuinely Iceberg. A Hive-format table will silently succeed at everything else.

### Step 1.6 — Tests, and the second target

**Why now.** "Tests in CI" is one of the six semantics, and CI is worthless if it needs the 3.5 GB CSV and a populated warehouse. (Note what that sentence does *not* say: the JVM is fine. Ubuntu runners ship Java, and a local-mode Spark session over a dozen fixture rows costs 2–3 minutes. What CI can't have is the *data*.)

**Write.**

- `unique` + `not_null` on `dim_article.article_id`, `dim_customer.customer_id`, and on the fact's grain (a `dbt_utils.unique_combination_of_columns`, or a hand-written singular test if you'd rather not add the package).
- `relationships` from `fact_transaction.article_id` → `dim_article.article_id`, same for customer.
- `accepted_values` on `sales_channel_id`.
- `seeds/`: ~200 hand-picked transaction rows, ~50 articles, ~50 customers, committed as CSV. Include at least one multi-quantity purchase and one null age, so the seeds exercise the cases the real tests catch.
- A `ci` target in `profiles.yml` using `dbt-duckdb`.
- `tests/test_iceberg.py::test_reload_is_idempotent` — the pytest wrapper §0 promised for `assert_identical`. It **drives** the property rather than observing it: create a temp table, load a day, capture the snapshot id, load the *same* day again, capture the new snapshot id, assert `assert_identical` on the two pinned reads. Marked `@pytest.mark.spark`.

**Must know — why the snapshot diff is not a dbt test.** The tempting version is a dbt singular test asserting "the newest two snapshots of `raw.transactions` differ by zero rows." It's wrong twice over, and both failures are instructive:

1. **It tests the last operation, not an invariant.** If the last thing you did was load a *new* day, the newest two snapshots *should* differ — a green result would mean the load did nothing. So the test passes when you re-ran and fails when you appended, which makes it a test of your recent shell history. dbt tests run on every `dbt build`; an assertion that's only meaningful right after a specific action doesn't belong there.
2. **And after step 1.3 it's awkward to write at all**, because it has to exclude `_ingested_at` — derivable in jinja via `adapter.get_columns_in_relation` plus a filter on the underscore prefix, but that's gymnastics in service of a test reason 1 already disqualified.

The general form of the lesson — **an idempotency check needs to control the action it's checking** — is why this is a test that *performs* two loads rather than a test that inspects a table. Assertions about a pipeline's behavior belong where you can drive the pipeline; assertions about a table's content belong in dbt. Getting that boundary right is most of what "tests in CI" means as a *semantic* rather than a checkbox.

**Must know — what CI actually proves.** The DuckDB target runs the same model SQL and the same tests against seeds. It does **not** prove your incremental strategy or your Iceberg write path — those are Spark-only, and they're what the pytest suite covers. Two consequences for how you write the models:

1. Keep the model SQL **dialect-portable**. Anything Spark-specific goes in `{{ config() }}` guarded on `target.type`, not in the body of the query. On the CI target, materialize everything as `table` and skip incremental entirely.
2. Say this in the README. "CI runs the dimensional model and its tests against fixture data on DuckDB, plus the point-in-time leakage test against a local Spark session, on every PR — the incremental and Iceberg write paths are exercised locally against the real warehouse" is a precise, credible claim. "My pipeline is tested in CI" is not.

**Checkpoint.** `dbt build --target ci` green in under a minute from a clean checkout with `data/` empty, and `pytest -m spark` green locally.

### Step 1.7 — GitHub Actions

**Mostly plumbing.** One workflow on `pull_request`: checkout → `actions/setup-python` 3.11 → `actions/setup-java` 17 → `pip install -e ".[dev]"` → `dbt build --target ci` → `pytest`. Thirty lines, not twenty, and the two extra things are load-bearing:

- **`setup-java` and the `.[dev]` install.** Without step 1.0's dependency block, `pip install -e .` installs *nothing*, so `import pyspark` fails and the PIT test errors rather than running. Without a JDK, it fails differently and more confusingly. These are the two reasons the obvious 20-line workflow goes red on first push for reasons that have nothing to do with your code.
- **A decision about the Spark tests.** Two defensible resolutions:
  - **Run them (recommended).** The spec and the timeline's gate 1 both say "dbt tests **+ the PIT-leakage test** on every PR" — the PIT test *is* the gate criterion, so a CI job that skips it isn't enforcing the gate. Cost is ~2–3 min of JVM startup on a job that would otherwise take 40 seconds.
  - **Skip them** with `pytest -m "not spark"` and run them locally via a `make test` target. Cheaper, faster, and it means gate 1's criterion is enforced by your discipline rather than by the machine. If you take this route, don't claim the PIT test runs in CI.

Take the first. The whole argument for CI here is that the PIT test runs when you *don't* remember to run it.

**Must know:** nothing technical. But make the PR discipline real — branch, PR, merge — because "tests run on every PR" is only true if you open PRs. A green badge on a repo where everything lands on `main` directly is a claim you can't defend.

**Checkpoint.** A PR with a deliberately broken model shows a red check, and the job log shows the Spark tests actually ran rather than being collected and skipped. Watch it fail once.

**→ Gate 1 is not passed yet.** The leakage test in week 2 is the gate criterion.

---

## 2. Week 2 — the feature pipeline (the centerpiece)

This is the week the project is judged on. Everything downstream inherits its correctness, and a PIT bug found in week 7 invalidates every number produced after now.

### Step 2.0 — The dataset fact that shapes everything here

**`t_dat` is a DATE. There are no intra-day timestamps.**

So "computed as of event time" is not literally available. The honest PIT rule for this dataset is:

> Every feature for an event on day `d` is computed over events in `[d − w, d − 1]`. Same-day events are excluded entirely.

Not "excluded because it's convenient" — excluded because with date-only granularity you cannot order two same-day events, so *any* same-day inclusion leaks an unknowable amount of future. Excluding the whole day is the only defensible rule, and it costs you real signal (same-day basket context is predictive). State the tradeoff in the README next to §2's other limitations. An interviewer who knows this dataset will ask, and "I used a strictly-prior window because the timestamps are date-only" is the answer that lands.

**And the follow-up question to that answer: the dimensions aren't point-in-time either.** Everything you compute from the event stream is PIT-correct. Nothing you read from `articles.csv` or `customers.csv` is — those are **snapshots as of download**, with no history and no valid-from/valid-to. So `dim_article`'s `product_type_no` and `dim_customer`'s `age`, `club_member_status`, and `fashion_news_frequency` are attached to 2018 events at their 2020 values. Two places this touches directly: `daily_cross_agg` (step 2.2) joins `dim_article` to get the category, and the customer tower (step 3.2) uses the customer attributes.

This is **unavoidable with this dataset and genuinely fine** — the attributes are mostly slow-moving or static, and there is no version history to use even if you wanted it. What is not fine is leaving it unsaid, because "your features are point-in-time correct" and "your dimensions are a current-state snapshot" are both true and the second one qualifies the first. Put it in the README's limitations list next to the `d − 1` rule, and say which attributes you'd expect to actually drift (`age` mechanically, `club_member_status` behaviorally) versus which are effectively immutable (a garment's product type). That distinction is what makes it a considered limitation rather than a discovered one.

If you want the full treatment as a stretch: a Type-2 slowly-changing dimension is the standard answer, and being able to describe how you'd build one — and why this dataset can't support it — is worth more than an hour of trying.

### Step 2.1 — Write the leakage test first, and watch it fail

**Why now.** Because it's the gate, and because a test written after the pipeline tends to test what the pipeline does.

**Write** `tests/test_pit.py`, two tests, both against a tiny hand-built DataFrame (a dozen rows, three customers) on a local Spark session:

1. **`test_window_excludes_same_day`** — a customer with purchases on days 10 and 11. Assert the 7-day count attached to the day-11 event is exactly 1. Not 2.
2. **`test_future_events_do_not_change_past_features`** — the strong one. Compute features over data truncated at day `D`. Compute them again over the full data. Assert the feature rows for days ≤ `D` are **identical**, column for column.

**Must know.** Test 2 is the one that matters and it's worth understanding why it's stronger than test 1. Test 1 checks a boundary you wrote deliberately, so it catches an off-by-one. Test 2 checks a *property* — "the past is a function of the past" — and it catches an entire class of bugs you didn't anticipate: a global mean imputation, a `StringIndexer` fit on the full dataset, a percentile computed over all time, a join that pulls a customer's lifetime total. Those are the leaks that survive code review, because none of them look like a window bug.

Test 2 also happens to be nearly the definition of a correct feature pipeline, which is why it's the gate.

**Checkpoint.** Both tests fail with a clear message, because the function under test doesn't exist yet. Commit them failing (or `xfail`ed) so the git history shows the order. That history is a small, real credibility signal.

### Step 2.2 — The daily aggregate layer

**Why now.** Rolling windows over 32M raw rows recomputed per event is the shape that explodes. Rolling windows over ~pre-aggregated daily rows is cheap and exact. This step is why the whole thing is tractable.

**Write** `src/marketrank/features.py`:

- `day_index(col)` — `datediff(t_dat, DATE'2018-09-20')` cast to int. One integer per calendar day.
- `daily_customer_agg(spark)` → `(customer_id, feature_date, day_index, n_txn, n_articles, spend, avg_price)`
- `daily_article_agg(spark)` → `(article_id, feature_date, day_index, n_txn, n_customers, avg_price)`
- `daily_cross_agg(spark)` → `(customer_id, product_type_no, feature_date, day_index, n_txn)` — joins `dim_article` in to get the category (see step 2.0 on why that join is not PIT).

**Carry both the date and the index, all the way through.** `day_index` is what the window frame orders on; `feature_date` is what the Iceberg `days(...)` partition spec needs in step 2.4, and it's what makes the table readable by a human or by dbt. Dropping the date at the aggregate layer and trying to reconstruct it at the write layer is a `date_add` on every row plus a chance to get the epoch wrong.

**Must know.** Why an integer day index and not the date. Spark's **range frames** (`rangeBetween`) require a single numeric ordering expression — they compare *values*, not row positions. That is exactly what you want and it's the difference between correct and nearly-correct:

- `rowsBetween(-7, -1)` = "the previous 7 **rows**." For a customer who shopped 7 times in two years, that's a two-year window. Silently wrong, and the features look plausible.
- `rangeBetween(-7, -1)` = "day_index within [d−7, d−1]." Correct regardless of gaps.

**Checkpoint.** Row counts per grain, and a spot-check on one heavy customer where you compute the same number by hand in the notebook.

### Step 2.3 — The rolling windows, and where PIT correctness physically lives

**Write** `rolling_features(daily_df, partition_cols, windows=(7, 30, 90))`:

```python
w = Window.partitionBy(*partition_cols).orderBy("day_index").rangeBetween(-days, -1)
```

Loop the three window lengths × the measures, `.over(w)`.

**Must know — this is the smallest, most important detail in the project.** The upper bound is `-1`. Change it to `0` and you have a leak: every feature includes the event's own day, the ranker's AUC jumps, and nothing else in the system tells you anything is wrong. The bug is one character. That is precisely why §10 calls PIT bugs "silent and flattering," and precisely why test 2.1 exists.

Say this out loud once so it's yours: *point-in-time correctness in this pipeline is enforced by the window frame, not by the join.* Because the features are already stamped "as of end of day d−1," attaching them to an event on day `d` is a **plain equi-join on (entity, day_index)** — no as-of join machinery, no interval logic, no correlated subquery. The expensive semantics got pushed into the frame, and what's left is cheap. That's the design insight of the week, and it's the one to lead with when someone asks how the feature pipeline works.

**Second thing you have to handle: the spine.** Window functions only produce output for rows that exist. A customer with no transaction on day `d` has no row on day `d`, so there is nothing to attach features to — but week 4 needs features for exactly those (customer, day) pairs, because candidates are scored on days the customer didn't buy the candidate.

The pattern: build a **spine** of every (entity, day) pair you will ever need features for, left-join it into the daily aggregate with zero-filled measures, window over the union, then filter back to the spine. Get this wrong and features exist only where labels are positive — which is its own, extremely flattering, kind of leak.

**Plumbing.** The measure-name mangling (`cust_n_txn_7d`, `cust_spend_30d`) and the loop that generates them.

**Checkpoint.** Both leakage tests pass. **This is gate 1.** Do not start week 3 until they do.

### Step 2.4 — Persist, split, backfill

**Write.**

- `feature_customer_daily`, `feature_article_daily`, `feature_cross_daily` as Iceberg tables carrying both `feature_date` (DATE) and `day_index` (INT), partitioned by `days(feature_date)`, written with partition overwrite.
- `build_features(spark, start=None, end=None)` — the **same function** does the full build and the backfill. A separate backfill script is the anti-pattern §5 is warning about; "recompute an arbitrary past window without a bespoke script" means the parameter, not the script.
- **The split constants, all six of them** (below). Central module, one place.

**The split.** Six disjoint slices, not four — because three different things downstream each need data the others haven't touched:

| Slice | Dates | Consumed by |
|---|---|---|
| `train` | ≤ 2020-08-11 | two-tower (wk 3), ranker (wk 5) |
| `val_tune` | 2020-08-12 – 08-25 | ranker early stopping / hyperparameters (wk 5) |
| `val_calib` | 2020-08-26 – 09-01 | isotonic calibration fit (wk 5.3) |
| `ope_env` | 2020-09-02 – 09-08 | **week 8's reward model / environment** |
| `test` | 2020-09-09 – 09-15 | reported NDCG, AUC, revenue numbers |
| `holdout` | 2020-09-16 – 09-22 | local MAP@12 sanity check (wks 3–5) |

Adjust the boundaries; keep the last 7 days as `holdout` so it mirrors the Kaggle test week's shape.

**Must know — why `val` had to be split in two, and why `ope_env` has to exist now.** Both are the same mistake in different places: a slice used twice tells you less than you think it does.

- **`val_tune` vs `val_calib`.** Fitting isotonic regression on the slice you also early-stopped on is mildly optimistic — the calibration map is fit to data the model was already tuned against, so measured ECE flatters itself. Not catastrophic, and plenty of production systems do it. But it's a free fix at this stage and an awkward caveat later.
- **`ope_env`.** Week 8's module is explicit: the reward model that serves as ground truth must be fit on **a different held-out slice than the ranker trained on**, or DM flatters itself and DR inherits the flattery through the residual term. That's rung 4 of the module's risk list. The failure mode if you skip it is not an error — it's a set of estimator-accuracy numbers that look great because the environment and the policy learned the same idiosyncrasies. Allocating the slice costs one line **now**; re-cutting data in week 8 means retraining the ranker to keep the boundaries honest.

Both of these are what "plan the slice split in week 2" means. This table is that plan.

**Must know — the backfill contract, and the trap inside it.** If day `D`'s events change, which feature days are invalid?

Not `D` itself. By the `rangeBetween(-w, -1)` rule, day `D`'s own feature row covers `[D − w, D − 1]` and **never sees day `D`**. The invalid range is `[D + 1, D + 90]` — the 90-day window on day `D + 90` still reaches back to `D`. So the contract is: *given changed source days `[a, b]`, recompute feature days `[a + 1, b + 90]`.*

(Recomputing `[a, b + 90]` instead is harmless — one wasted day. It's stated exactly here because a document whose thesis is that the `-1` bound is the most important character in the project should not be off by one in its own contract.)

**And the trap the contract hides — name it, because it's the same shape as `rowsBetween`.** `build_features(start, end)` must **read** source data back to `start − 90` and only **write** `[start, end]`. If the implementation filters the daily aggregates to `[start, end]` *before* windowing — which is the natural way to write it, and which looks like an obvious optimization — then every recomputed window is truncated at the left edge. A backfilled day near `start` gets a 90-day feature computed from 3 days of data. No error, no null, just a number that's quietly wrong, in a table that already passed its leakage tests.

Call it the **backfill-truncation trap** and put it in the function's docstring next to the range rule: *read `[start − max_window, end]`, write `[start, end]`.* The asymmetry between the read range and the write range **is** the function's contract; everything else is detail.

**Checkpoint.** Recompute an arbitrary 30-day window mid-2019 and diff it against what the full run produced — identical. That single diff catches the truncation trap, the off-by-one, and any non-determinism, which is why it's the checkpoint. Record the wall clock and rows processed; §7's metric table asks for both, and week 4's single-node-vs-multi-node decision is made from this number.

---

## 3. Week 3 — retrieval

### Step 3.1 — Baselines first

**Why now.** Because on H&M the naive baselines are *strong*, and a two-tower model that doesn't beat them is a two-tower model you shouldn't ship. Finding that out in week 3 is cheap; finding out in week 6 is not.

**Write** `src/marketrank/retrieval/baselines.py`: top-N by recent popularity, and **repurchase** (the customer's own previously-bought articles, most-recent-first). Measure recall@100 and recall@500 on `val_tune` (step 2.4's table — retrieval has no separate calibration or environment need, so it uses the tuning slice).

**Must know.** Repeat purchase is a large fraction of H&M's signal, and the popularity baseline is what the competition's median entry effectively was. Your two-tower's job is to beat *the union of both*, not to beat random. Write the baseline numbers down — they're the denominator for "recall@__ up __% over baseline" in §11, and that bullet is a lie if the baseline was chosen to be beatable.

### Step 3.2 — The two towers

**Write** `src/marketrank/retrieval/model.py` — PyTorch, reading Iceberg directly.

- **Article tower:** an id embedding (105k × d) plus embeddings for `product_type_no`, `colour_group_code`, `department_no`, `index_group_no`, `garment_group_no`; concatenate, MLP to d=64 or 128, L2-normalize.
- **Customer tower:** `age` (bucketed), `club_member_status`, `fashion_news_frequency` — all three are current-state snapshot attributes, not PIT, per step 2.0 — plus the week-2 rolling features, plus, the important one, a **mean of the embeddings of the customer's recent articles**, taken from the PIT feature window.
- Score = dot product.

**Think first.** Why include the recent-item-average in the customer tower rather than just a 1.37M-row customer id embedding?

<details>
<summary>Answer</summary>

Three reasons, and you should be able to give all three: (1) a 1.37M × 128 id embedding is 175M parameters for a table where most customers have a handful of transactions — it memorizes and doesn't generalize; (2) it cannot serve a customer who wasn't in training, so cold start is unsolvable by construction; (3) the item-average version updates the moment a customer buys something, without retraining, which is what makes the serving path in week 6 honest. Real production two-towers are built this way for exactly these reasons.

And note the PIT constraint carries through: "recent articles" means recent **as of day d−1**. Same rule, new place. This is where a leak sneaks back in if you're not watching.
</details>

**Must know — sampled softmax and the logQ correction.** In-batch negatives means every other positive in the batch acts as a negative. That's efficient and it's what everyone does. It's also **biased toward popularity**: popular articles appear in more batches, so they're penalized as negatives more often, and the model learns to under-rank them. The fix is one term — subtract `log(sampling_prob(article))` from the logits before the softmax. It's three lines and it's the single most interviewable detail in this week. Know why it's there, not just that it's there.

**Plumbing.** The training loop, the dataloader, checkpointing.

**Checkpoint.** Recall@100 on `val_tune` beats the baseline union by a margin you'd defend. If it doesn't, the problem is almost always negatives or the item-average feature, not the architecture.

### Step 3.3 — The ANN index

**Write.** hnswlib or `faiss.IndexFlatIP` over the 105k article vectors; `retrieve(customer_vec, N)`.

**Must know — and be honest about it.** 105k vectors at d=128 is 54 MB. Exact inner-product search over it takes single-digit milliseconds. **ANN is not needed here for speed**, and claiming otherwise is the manufactured-scale trap wearing a different hat — the same trap §10 warns about for Spark.

The defensible framing: build both, measure both, and report the recall-vs-latency tradeoff at this catalog size, then say what changes at 10M items. "At 105k items exact search is 4 ms and HNSW is 0.4 ms at 98% recall — the index earns its place in the serving path's tail latency, not in its median" is a *better* answer than pretending you needed it. Same move as "I used Spark because of the join fan-out."

---

## 4. Week 4 — candidate generation

The one genuinely cluster-shaped job. Budget the week; the spec says so and it's right.

### Step 4.1 — Candidate sources

**Write** `src/marketrank/candidates.py`. For each (customer, day) in the training spine, generate N≈100 candidates from a **union** of sources, each tagged with its origin:

- two-tower ANN top-k
- the customer's own repurchase set
- top popular in the customer's dominant category

**Must know.** Two things.

*Tag the source.* "Which source did this candidate come from" is a feature the ranker uses and it's also your retrieval diagnostic — if 90% of the ranker's top-12 come from repurchase, your tower isn't contributing and you need to know that before week 6's baseline comparison.

*Recall ceiling.* Compute the fraction of true purchases that appear anywhere in the candidate set. That number is the **hard ceiling on end-to-end recall** — the ranker cannot recover a purchase that stage 1 dropped. It's the honest measure of whether stage 1 is doing its job, and it's the number that makes the two-stage argument concrete rather than architectural.

### Step 4.2 — The join, and the size

**Write.** Join candidates → PIT features on `(customer_id, day_index)` and `(article_id, day_index)` and the cross grain. Write Iceberg, partitioned by day.

**Must know.**

- **Sizing.** positives × N × (feature width × bytes) — do this arithmetic *before* you run it, from your own row counts, and write the number in the README. §5 says 40–160 GB; your actual number depends on how much history you use as positives. The point of computing it is that "I used Spark because the join fan-out is ~__ GB" requires a `__` you measured.
- **Skew.** Heavy customers and popular articles make hot partitions. Salt the article-side join key or broadcast the article features (they're small) — broadcast is usually the answer here, and it's worth knowing *why* and *that it won't happen by itself*: `spark.sql.autoBroadcastJoinThreshold` defaults to **10 MB**, and the article dimension is ~100 MB, so Spark will plan a shuffle hash join unless you say otherwise. Either wrap the side in `F.broadcast()` or raise the threshold deliberately. The mechanism worth being able to state: broadcasting ships one copy of the small side to every executor so the join happens map-side with **no shuffle of the 100+ GB side at all** — which is the entire cost of this job. Also know the failure mode you're trading into: the broadcast side materializes in the driver and then in every executor's heap, so pushing the threshold up far enough turns a slow join into an OOM.
- **Negative downsampling and the calibration debt it creates.** You will not keep all ~100 negatives per positive. If you sample negatives at rate `w`, the model's output probability is no longer the real-world probability, and the decision layer in week 7 consumes it as one. The correction is Elkan's: `p = p_s / (p_s + (1 − p_s)/w)`. **Write down `w` now.** A forgotten sampling rate is the most common way a calibrated-looking ranker turns out to be wrong by a constant factor, and every revenue number in week 7 inherits it.

**Checkpoint.** Re-run the leakage property test (2.1's test 2) **on the joined candidate table**, not just on the feature tables. The join is a new opportunity to leak. Plus row-count and null audits per §8.

### Step 4.3 — misha

Follow [`SETUP_MISHA.md`](SETUP_MISHA.md). Per the timeline: **measure single-node first.** 64 cores and 480 GiB may absorb this entire job, and "480 GiB single-node Spark with tuned partitioning and spill" is the stronger claim when it's the true one. Go multi-node only if the single-node wall clock is genuinely painful — and record both numbers if you do, because the comparison *is* the anecdote.

---

## 5. Week 5 — ranking

### Step 5.1 — Objective choice, which is really a calibration decision

**Think first.** LightGBM offers `lambdarank` (directly optimizes NDCG) and `binary` (logloss). The spec's gate requires both a good NDCG **and** a calibrated probability. Which objective, and why is this not a free choice?

<details>
<summary>Answer</summary>

`binary`. `lambdarank` optimizes *ordering* and its raw scores have no probabilistic meaning at all — they're not miscalibrated, they're not probabilities. You can still compute NDCG from a binary-objective model's scores (ranking only needs order, and a well-fit probability orders well), but you cannot recover probabilities from a lambdarank score without fitting a whole separate calibration map on held-out data.

Since the decision layer multiplies p̂ by price and treats the product as expected revenue, a score that isn't a probability corrupts every number in weeks 7 and 8. So: train `binary`, report NDCG@12 from it, and calibrate. If you want the comparison, train lambdarank too and show it ranks slightly better and calibrates worse — that's a nice half-page in the README and it makes the choice look like a decision instead of a default.
</details>

### Step 5.2 — Features and training

**Write** `src/marketrank/ranker.py`. Features: the PIT rolling aggregates, article metadata, customer attributes, candidate-source tags, and **cross features** — customer's historical rate in this product type, and `price / customer's average historical price`. That last one is not optional: it's the feature that connects the ranker to the decision layer, because price sensitivity is what week 7 acts on.

Groups for NDCG are `(customer_id, day)`.

**Must know.** Train on candidates, not on transactions. The ranker's job is to order *the distribution stage 1 produces*, so training it on a different distribution than it will see at serving time is the classic two-stage mistake. This is also why week 4 comes before week 5 rather than in parallel.

### Step 5.3 — Calibration, which is a gate criterion

**Write.** Reliability diagram (predicted bucket vs. observed rate), ECE, then isotonic regression fit **on `val_calib`** — not on train, not on `val_tune` (which the ranker was early-stopped against), not on test. That slice exists for exactly this; see step 2.4.

**Must know.** Three separate things push your probabilities off, and you should be able to name all three:

1. **Negative downsampling** (step 4.2) — a known multiplicative distortion with a closed-form correction.
2. **The candidate distribution** — p̂ is `P(purchase | in the candidate set)`, not `P(purchase)`. That's *fine* and arguably what you want, but it means the number is conditional and the README should say on what.
3. **Ordinary miscalibration** from the objective and regularization — what isotonic fixes.

Fix (1) analytically, state (2), fix (3) with isotonic. Reporting a calibration curve without knowing which of these you corrected is the shape of an answer that falls apart under one follow-up question.

**Checkpoint.** ECE before and after, both recorded. The curve goes in the README. **Gate 2's calibration criterion is this plot.**

---

## 6. Week 6 — connect, serve, and the one external number

### Step 6.1 — The FastAPI path

`GET /recommend?customer_id=...&k=12` → retrieve → hydrate features → rank → return, with per-stage timings.

**Must know — feature hydration is where training/serving skew lives.** At serving time you need the same feature vector training used, keyed by (customer, latest feature day). The spec rules out Redis, and rightly: read the latest feature partition into a DuckDB file or an in-memory dict at startup. One code path computes features (week 2), one table stores them, serving reads that table. If serving ever recomputes a feature with different code, you have skew, and skew is invisible offline.

Measure p50/p95/p99 per stage over a few hundred requests. Report the tail, not the mean.

### Step 6.2 — Two-stage vs. one-stage

Score the full 105k catalog with the ranker for a sample of customers, compare quality *and* latency against the two-stage path. This is v2's money chart, demoted to secondary — but it's the plot that proves stage 1 earns its existence.

### Step 6.3 — The Kaggle submission — **one day, hard stop**

Retrain on data ≤ the competition's training cutoff, emit top-12 per customer for every customer in `customers.csv`, submit once, record score **and rank**, disclose as a late submission everywhere it appears.

**VERIFY on the competition page:** the test week (2020-09-23 → 2020-09-29 by my reading of the data's end date), the submission format, and that late submission is still open. Confirm before you spend the day.

**The trap, restated because this is where it bites.** The feedback is fast and quantified and arrives exactly when weeks 7–8 don't exist yet. MAP@12 rewards ordering heuristics that do nothing for a calibrated ranker or a decision layer. Submit once. If you find yourself tuning for it, you're trading the half of this project nobody else has for the half 2,952 people already did.

---

## 7. Week 7 — the decision layer

### Step 7.1 — Price response, and the honest version of it

**The problem, stated plainly:** you observe prices paid on purchases. You do not observe a customer declining at a price. There is no experiment. Markdowns are timed against seasonality and inventory, so naive elasticity is confounded and will *overstate* responsiveness.

**Write.** Per (article, week): mean price and units sold. Regress `log(units)` on `log(price)` with **article fixed effects** (within-article variation only) and **week-of-year controls** (seasonality). The coefficient is ε̂.

Then: `p̂(purchase | d) = p̂₀ · (1 − d)^ε̂`, with ε̂ < 0 so a discount raises it. And `E[rev | d] = p̂(d) · price · (1 − d)`.

**Must know.** State the identifying assumption in the README — *conditional on article and season, remaining price variation is as-good-as-random* — and then say that you don't fully believe it. This is the **weakest causal claim in the project** and the spec instructs you to label it that way. Do not let week 7's revenue numbers inherit more confidence than ε̂ supports; that inheritance is exactly what week 8 exists to bound.

### Step 7.2 — Constrained allocation

Given a slate of k items with p̂ and price, choose a discount level per item from a small menu (0%, 10%, 20%, 30%) to maximize Σ E[rev | d] subject to Σ expected discount spend ≤ B.

**Write.** Greedy by marginal revenue per unit budget first — it's a knapsack, greedy is the natural heuristic, and it's rung 2 of the scope-cut ladder, so it must work standalone. Then LP via PuLP for the comparison.

**Must know.** Why this is a real constrained decision and not a threshold: the budget couples the items. Discounting the item with the best individual lift may consume budget that two other items would have used better. That coupling is the whole reason it's an optimization problem, and "I allocated under a budget constraint" only means something if you can say what the constraint made you give up.

### Step 7.3 — Guardrails

Name them and set thresholds *before* you see the results: catalog coverage, new/low-exposure article share, per-segment relevance, discount spend. State the level at which you'd refuse to ship or roll back.

**Must know.** A policy that lifts revenue by starving the long tail is a policy you should be able to catch. Metric design is heavily interviewed for product/marketplace DS and it costs you almost nothing here — but the thresholds are only credible if they were set before the numbers existed.

### Step 7.4 — The frontier, first pass

Sweep the price weighting; plot expected revenue against relevance, one point per policy. No confidence bands yet — those are week 8, and that's the point.

---

## 8. Week 8 — off-policy evaluation

Follow [`Project_D_v3_Module_OffPolicy_Evaluation.md`](../../brainstorm_project/Project_D_v3_Module_OffPolicy_Evaluation.md) day by day; it's already spec'd at implementation depth. Four things to carry in from this document:

1. **The reward model must be fit on a different held-out slice than the ranker.** Otherwise DM flatters itself and DR inherits the flattery. That slice is **`ope_env` (2020-09-02 – 09-08)**, allocated in step 2.4 — use it and nothing else, and check before you start that nothing in weeks 3–7 quietly borrowed it.
2. **The slate trap.** You serve k items; bandit estimators assume one action. Rung 2 of the module's ladder — evaluate the **top-1 decision only**, where the bandit assumption holds without argument — is the right default. Silently applying bandit estimators to a slate is the failure.
3. **π₀ must be stochastic.** A deterministic logging policy has zero support off its own argmax and OPE is simply impossible. Softmax over expected-revenue scores at temperature τ; τ is the divergence knob and the x-axis of the module's chart.
4. **No estimate ships without ESS and a bootstrap CI beside it.** Never-cut, and with revenue rewards — right-skewed, times heavy-tailed importance weights — also report max-weight share. If one observation carries most of the estimate, say the number isn't usable rather than rounding it off.

**Output:** CI bands onto the week-7 frontier. That's the money chart, and gate 3.

---

## 9. Week 9 — data quality and late-arriving data

### Step 9.1 — The late-arrival exercise

This is where step 1.2's deferred decision comes due. Replay a day's events out of order — split day `D` into two files, load the second one after `D+1` has already landed — and confirm the tables converge to the same state as a clean full load.

**Think first.** The obvious move is `MERGE INTO fact_transaction` on the step-1.4 grain, incrementing `qty` when the key already exists. What happens when the delta file gets replayed — a retried job, a re-run after a crash, a corrections file sent twice?

<details>
<summary>Answer</summary>

`qty` double-counts. An incrementing merge is **inherently non-idempotent**: its result depends on how many times it ran, which is the definition of what idempotency rules out.

That's worth sitting with, because of *which* two things collide. Semantic #3 (late-arriving data) would have been implemented with a mechanism that silently breaks semantic #1 (idempotent re-runs) — and #1 is the one you spent all of week 1 establishing and can currently prove. The demo would work: replay once, tables converge, screenshot it. It breaks on the *second* replay, which is exactly the case an interviewer probing "how do you handle late-arriving data" asks about, because it's the case that happens in production.
</details>

**Write — and name the layer, the builder, and the mechanism for each step.** This is the step where being vague about which component does what produces a demo that works once. You chose grain (b) in step 1.4, so `fact_transaction` is a **pure aggregate of `raw.transactions`** — which means the fix goes in at the source and propagates, rather than being applied to each table by hand:

| # | Layer | Built by | Mechanism | Idempotent because |
|---|---|---|---|---|
| 1 | `raw.transactions` | `ingest.py` (Spark) | append the corrections file, carrying `_ingested_at` and a new `_source_file` | the append is guarded — see 2 |
| 2 | `raw.transactions` | `ingest.py` | provenance guard: if that `_source_file` is already in the table, no-op | re-applying a known batch does nothing |
| 3 | `fact_transaction` | **dbt** | rebuild the model | it's a pure function of `raw` |
| 4 | `feature_*_daily` | `build_features(D+1, D+90)` | partition overwrite | re-running writes the same partitions |

`_source_file` is a second schema evolution, and this time you have a reason for it rather than the `promo_flag` exercise — same `ALTER TABLE ADD COLUMN`, same metadata-only cost (step 1.3). The 734 already-loaded partitions read back null there, which is exactly right: they predate provenance tracking, and a null never collides with a named batch, so the guard is correct on day one without a backfill.

**Must know — step 3 does *not* use the `overwritePartitions` you already have, and it's worth knowing why.** That call lives in `ingest.load_transactions`, which writes **`raw`**. `fact_transaction` is a dbt model one layer up, and dbt owns its write path. So "re-aggregate day `D`'s fact partition" needs a mechanism that exists at *that* layer, and there are only three:

- **(a) A dbt incremental model** with `incremental_strategy='insert_overwrite'` on the Spark target, partitioned by `t_dat`. This is the real answer at scale, and it's cheap to add. Its one honest caveat: step 1.6's CI target materializes everything as `table` on DuckDB, so the incremental path is verified locally and never in CI. Say that rather than implying otherwise.
- **(b) A Python/Spark job that rebuilds the day and writes the table directly.** Don't. That's a second builder for a dbt-owned table — the same skew pattern step 6.1 warns about in serving, one layer down. Two code paths that must agree forever, and nothing tells you when they stop.
- **(c) Just rebuild the whole model.** `dbt build --select fact_transaction`. At 31M rows this is a couple of minutes on Spark.

**Take (c) now, and put (a) in "what changes at 100×."** That's the honest ordering: at this data size a full rebuild is cheaper than the machinery that avoids it, and knowing the exact size at which that flips is a better answer than having built the machinery prematurely. If you want (a) anyway it's an afternoon — just don't claim CI covers it.

**Must know — where the `MERGE` artifact goes, if you want one.** `MERGE INTO` is one of Iceberg's selling points and §5 names it as the mechanism behind idempotency, so it's reasonable to want it in the repo. Two coherent placements. Pick one:

- **On `raw`, keyed on provenance.** The step-2 guard expressed as `MERGE` instead of an existence check: `ON t._source_file = s._source_file`, `WHEN NOT MATCHED THEN INSERT *`. If the batch already landed, every source row matches something and nothing inserts; if it didn't, nothing matches and all rows insert. Fact stays derived. (No `WHEN MATCHED` clause, so the multiple-match cardinality check doesn't apply — **VERIFY** on your Iceberg version.)

  **Keep the existence check as the operational guard and treat this as the artifact.** The join key here is *batch-constant*, so a replayed batch of `n` rows joins to all `n` existing rows with that `_source_file` — `n²` pairs evaluated to conclude "do nothing." At 3 rows that's noise; replay a 40k-row day and it's 1.6 billion pairs to reach a no-op. The `SELECT 1 ... WHERE _source_file = ? LIMIT 1` version answers the same question in one predicate. Being able to say *why* the elegant form is the wrong one to put in the loop is worth more than either implementation.

- **On `fact`, keyed on the grain alone, sourced from the re-derived full day.** `WHEN MATCHED THEN UPDATE SET *`, `WHEN NOT MATCHED THEN INSERT *`. Idempotent and grain-preserving. Route it through dbt's `incremental_strategy='merge'` with `unique_key` set to the grain, and it stays a single builder.

  **Then pick which of the two virtues you want, because they don't come together.** dbt's merge strategy generates only matched-update and not-matched-insert from `unique_key` — there is **no way to express `WHEN NOT MATCHED BY SOURCE ... DELETE` through it**, so the dbt-routed version cannot remove rows that vanished from a corrected day. For this exercise that costs nothing: the corrections are append-only, so no row ever vanishes. If you want the full reconcile — `WHEN NOT MATCHED BY SOURCE AND t_dat = 'D' THEN DELETE`, which is the better demonstration of what `MERGE` is actually *for* — it has to be hand-written SQL, and then be explicit that it's **a documented one-off for the demonstration, not a second standing builder**. That keeps option (b)'s trade visible instead of re-entering it quietly. (The clause needs Spark 3.5 + Iceberg ≥ 1.5 and you're on 1.11.0, so it should be available — **VERIFY**, same as the cardinality question above.)

**And the key trap, since it's the one that looks right.** Do **not** key the fact-level merge on `grain + _source_file`. It reads like the safe conservative choice — provenance in the key, nothing can collide — and it silently destroys the grain: a corrected row carries a *different* `_source_file` than the original, so it never matches, always inserts, and you end up with two rows per basket line with `qty` split across them. Step 1.6's uniqueness test on the (b) grain goes red the first time a correction lands, and if you'd skipped that test you'd have found out in week 7 when revenue numbers stopped adding up. Provenance belongs in the key at `raw`, where the grain is "a row as delivered." At `fact`, the key is the business grain and nothing else — that's what makes it a grain.

State the choice in the README either way. "The fact table is an aggregate of the raw log, so late arrivals are handled by correcting the log and re-deriving rather than by patching the aggregate — which is idempotent without needing a delivery guarantee" is a strong sentence, and it's the kind of reasoning the six-semantics framing is asking for.

**Must know — the base loader has to be redefined, or it will eat your correction.** Step 3's guarantee is "a full-day refresh from the source of truth is idempotent." The word doing the work is *source of truth*, and after a correction lands, that is no longer the base CSV. `load_transactions(spark, "D", "D")` as it stands is an unguarded partition overwrite reading `transactions_train.csv` alone — run it after a correction and the correction is silently gone. Worse, `assert_identical` will then cheerfully confirm that the wrong state is stable, because stability is all it tests.

So redefine it, in the docstring and in the code: **a full-day refresh reads every batch for that day — the base extract plus any corrections files — unions them, and overwrites the partition.** The source of truth is a *set of files*, which is precisely why `_source_file` exists as a column rather than as a log line. Get this wrong and semantic #3 undoes itself using semantic #1's own mechanism, which is a genuinely interesting way to fail and not one you want to discover live.

**Must know — the part that makes this more than a re-run.** Fixing the fact table is half the job. Because of step 2.4's window contract, changed source day `D` invalidates feature days `[D + 1, D + 90]` — so the handler ends with `build_features(start=D+1, end=D+90)`, and that call reads back to `D + 1 − 90` per the truncation rule. Demonstrating that chain is the strongest single thing in this week, and it only exists because you wrote the contract down in week 2.

**Checkpoint.** Three assertions, one test:

1. After the corrections file lands, the tables match a clean full load.
2. Replay the **same** corrections file — the tables are **unchanged**. This is what separates "converges once" from "converges."
3. Re-run the plain full-day load for day `D` afterwards — the correction **survives**. This is the one that fails if you skipped the redefinition above, and it fails silently.

### Step 9.2 — Freshness and anomaly tests

dbt `source freshness` on `_ingested_at` (this is what step 1.3 bought you), plus row-count anomaly tests per day, plus a drift check on the served path.

**Use the real outlier, not a synthetic one.** Step 1.4's count found a (customer, article, day) group with **570 rows**. Write the test so it would flag that — a threshold on units-per-customer-per-article-per-day — and you have a data-quality check that catches something that actually exists in your data rather than a rule invented to have a rule. Then decide what the pipeline *does* about it: exclude it from training as non-organic, cap it, or keep it and say why. That decision is the interesting half; the test is the easy half.

---

## 10. Week 10 — writeup and buffer

The README carries: the three layers; §2's limitations **verbatim** (single retailer, scaled prices, no logged experiment); the PIT argument; the money chart; measured numbers where the specs have `__`; "what changes at 100×."

Add the three limitations this build discovered, in the same list and the same plain voice — they're §2's genre, found by doing the work rather than by reading the spec:

- **Features are computed over `[d − w, d − 1]`** because `t_dat` is date-only, so same-day context is excluded entirely (step 2.0).
- **Dimension attributes are current-state snapshots**, not point-in-time, and there is no version history in the dataset to fix that (step 2.0).
- **Elasticity is identified off observational within-article price variation**, and is the weakest causal claim in the project (step 7.1).

Then the **"How I would A/B test this policy"** section — a design, not a build. Randomization unit (customer, not session — interference), power calculation from your own observed revenue variance, week-7's guardrails as decision criteria, stopping rule, rollback trigger. And explicitly: why OPE was the available tool here and an experiment was not.

Per the spec, this section is what converts the experimentation résumé variant from adjacent to direct — the project contains no A/B test, and an experimentation interviewer will ask about power, peeking, variance reduction, and interference, none of which OPE covers.

Then §11's bullets, with every `__` filled from a number you measured.

---

## Reference — dataset facts, so you don't re-derive them

| Fact | Value | Consequence |
|---|---|---|
| `transactions_train.csv` span | 2018-09-20 → 2020-09-22, 734 days | Loaded, all partitions present |
| Source rows | **31,788,324** | Fact table is 28,583,889 — a 10.1% collapse |
| `articles.csv` / `customers.csv` rows | **105,542** / **1,371,980** | The résumé's "105k / 1.4M" |
| …that ever transact | **104,547** / **1,362,281** | 995 unsold articles, 9,699 silent customers — cold start (wk 3), and `relationships` only holds fact→dim |
| `t_dat` granularity | **DATE, no time** | PIT window must end at `d − 1` (§2.0) |
| `article_id` | zero-padded string | Never let Spark infer it |
| Duplicate transaction rows | real — 9.51% of (customer, article, day) groups | Drives the grain decision (1.4) |
| Same group, differing price | **0.80%** of groups, 97.9% not a channel effect | Price is a measure, not a key (1.4) |
| Largest single group | **570 rows** | Not organic — week 9 anomaly-test case |
| `price` | scaled, not currency | All revenue results are relative, per spec §2 |
| `customers.csv` | `FN`/`Active` sparse, `age` nullable | Cast at staging, not at read |
| `articles.csv` / `customers.csv` | **current-state snapshots, no history** | Dimension attributes are not PIT (§2.0) — README limitation |
| Kaggle test week | 2020-09-23 → 2020-09-29 (**VERIFY**) | Week 6 retrain cutoff |

## Reference — target repo layout

```
src/marketrank/
  config.py  spark.py  ingest.py  checks.py      # exists
  features.py                                     # week 2
  retrieval/  model.py  baselines.py  index.py    # week 3
  candidates.py                                   # week 4
  ranker.py  calibration.py                       # week 5
  decision.py                                     # week 7
  ope/  estimators.py  logging_policy.py  env.py  # week 8
  serve/  app.py                                  # week 6
dbt/     models/staging  models/marts  seeds      # week 1
tests/   test_pit.py  test_iceberg.py             # weeks 1–2
conf/    spark-defaults.conf                      # week 1
Makefile                                          # grows every week
```

## Decisions this document defers to you

1. ~~**Fact grain** (step 1.4)~~ — **settled 2026-08-14 by measurement.** Price stays out of the key; it's a measure. See step 1.4 for the numbers and the one-line argument.
2. **Embedding dimension and customer-tower composition** (3.2) — pick after the baseline numbers exist.
3. **How much transaction history becomes training positives** (4.2) — this single choice sets whether candidate generation is 40 GB or 160 GB, and therefore whether misha is single-node or multi-node.
4. **Discount menu and budget level** (7.2) — arbitrary, so pick round numbers and say they're illustrative.
