# marketrank

Retail personalization + promotion allocation on the H&M Personalized Fashion
Recommendations dataset: an Iceberg/dbt lakehouse, a point-in-time-correct Spark
feature pipeline, two-stage retrieval and ranking, a budget-constrained pricing
decision, and off-policy evaluation of that decision.

Build log with every measured number: [`BUILD_NOTES.md`](BUILD_NOTES.md).

---

## Honest framing of the dataset

Verbatim from the project spec, because an interviewer will find these:

- **H&M is a single retailer, not a two-sided marketplace.** There are no
  sellers, no supply side, no matching problem. The honest description is
  **retail personalization + promotion allocation**.
- **Prices are scaled, not currency.** Every revenue figure is in arbitrary
  units, so all revenue results are **relative** — "+x% expected revenue at
  equal relevance", never a dollar amount.
- **There is no logged experiment and no logging policy.** Transactions are
  what the existing merchandising surfaced, with no recorded propensities.

## Limitations discovered by building it

- **Features are computed over `[d − w, d − 1]`.** `t_dat` is date-only, so two
  events on the same day cannot be ordered and *any* same-day inclusion leaks an
  unknowable amount of future. Same-day basket context is excluded entirely, and
  that costs real signal.
- **Dimension attributes are current-state snapshots, not point-in-time.**
  `articles.csv` and `customers.csv` have no history and no valid-from/valid-to,
  so a 2018 event is joined to 2020 attribute values. `age` drifts mechanically
  and `club_member_status` behaviourally; a garment's `product_type_no` is
  effectively immutable. A Type-2 slowly-changing dimension is the standard fix
  and this dataset cannot support one.
- **Elasticity is identified off observational within-article price variation**
  and is the weakest causal claim in the project.

---

## Data model

### The fact grain

One row of `fact_transaction` is **one basket line**:
`(customer_id, article_id, t_dat, sales_channel_id)` with a `qty` count.

Measured on the full 31,788,324-row log:

| Quantity | Value |
|---|---|
| Source rows | 31,788,324 |
| Distinct `(customer, article, day, channel)` | 28,583,889 |
| Distinct `(customer, article, day, channel, price)` | 28,813,419 |
| Groups carrying more than one distinct price | 223,068 (0.78%) |
| Max distinct prices within one group | 8 |
| **Multi-quantity purchase rate** | **10.08%** of source rows collapse |

The key is a function of the business grain, not of row order. A surrogate key
from `row_number()` or `monotonically_increasing_id()` would depend on
partitioning and read order, which Spark does not guarantee across runs — so
re-loading a day would produce "the same data" with different keys and every
downstream incremental merge would see every row as new.

**`price` is not in the key.** Only 0.78% of basket lines saw more than one
price, and a mid-day markdown is not a different basket line. The fact table
carries `price_mean = sum(price)/qty`, plus `price_min` and `price_max` so the
markdown is still visible, and `revenue = sum(price)` — which is exact and
equals `qty * price_mean` regardless of within-group price variation. Keeping
`price` out of the key also removes the float-key fragility entirely; for the
record, it would have been safe here anyway, since all 9,857 distinct prices
survive a cast to `DECIMAL(10,8)` without collision.

### The modeling layer

```
local.raw.transactions   Iceberg, partitioned by days(t_dat), 734 days
local.raw.articles       Iceberg, unpartitioned snapshot
local.raw.customers      Iceberg, unpartitioned snapshot
   |  dbt
   v
staging.stg_*            ephemeral (rename + cast, no business logic)
   |
   v
local.marts.fact_transaction / dim_article / dim_customer   Iceberg tables
```

Staging models are **ephemeral**, not views: Iceberg's `SparkCatalog` does not
implement `ViewCatalog`, so `create or replace view` inside the `local` catalog
fails with *"Replacing a view is not supported by catalog: local"*. Since staging
here is pure projection, inlining it as a CTE stores nothing and keeps the dev
and CI targets running the same thing.

### What CI actually proves

CI runs the dimensional model and its tests against fixture data on DuckDB, plus
the Iceberg idempotency tests and the point-in-time leakage test against a local
Spark session, on every PR. The **incremental and Iceberg write paths are
exercised locally against the real warehouse**, not in CI — CI has a JVM but no
data, so the seeds stand in for the 3.5 GB extract.

That is the precise claim. "My pipeline is tested in CI" is not.
