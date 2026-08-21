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

---

## Measured numbers (reduced scale where marked)

All produced by commands in this repo. Full provenance in
[`BUILD_NOTES.md`](BUILD_NOTES.md); anything not measured says "not run" there
rather than carrying an estimate.

### Data layer — full scale

| | |
|---|---|
| Transactions loaded | 31,788,324 over 734 days (2018-09-20 → 2020-09-22), 49.3 s |
| Articles / customers | 105,542 / 1,371,980 |
| `fact_transaction` | 28,583,889 basket lines; multi-quantity rate 10.08% |
| dbt on Spark/Iceberg | 16 models + tests green in 2 min 09 s |
| dbt on DuckDB (CI) | 19 nodes green in 4.5 s, no warehouse, no CSV |
| Feature tables | 9,080,179 + 7,443,545 + 19,980,389 rows, full build 367.8 s |
| 30-day backfill | 51.5 s, 1,960,023 rows, bit-identical to the full run |

**Gate 1 (point-in-time leakage) passed**, and both tests were mutation-checked:
flipping the window's upper bound from `-1` to `0` fails test 1; a global mean
fitted over all time fails test 2.

### Retrieval — `val_tune`, 20,000-customer cohort, 70,715 true pairs

| | recall@12 | recall@100 | recall@500 |
|---|---|---|---|
| repurchase | 2.32% | 3.31% | 3.39% |
| recent popularity | 1.22% | 6.25% | 17.98% |
| baseline union | 2.51% | **6.97%** | 18.99% |
| two-tower (reduced scale) | 1.21% | **5.53%** | 15.15% |

**The two-tower does not beat the baseline union**, so step 3.2's checkpoint is
recorded as failed rather than tuned until it passed. The `logQ` correction is
worth 17x on recall@500 (11.22% with, 0.67% without) — and the run without it has
the *lower* training loss.

Candidate set (three sources, ~105 per customer): **recall ceiling 7.475%**.
That is a hard cap on end-to-end recall, and it says stage 1 is this build's
bottleneck.

### ANN, at this catalog size

105,543 vectors × 64 dims = 27.0 MB. Single-query exact search is 1.26 ms p50 /
1.82 ms p95; HNSW at ef=200 is 0.16 ms p50 / 0.27 ms p95 at 98.6% index recall.
**ANN is not needed here for speed** — it earns its place in the serving path's
tail latency, not in its median, and at 105k items not in the architecture story
at all.

### Candidate-generation sizing

Candidates are generated once per **(customer, day)**, not once per positive —
a three-article basket is one scoring event with one candidate set. Measured on
the `train` slice: 27,155,032 positives across **8,584,379 scoring events**, a
mean basket of 3.16.

8,584,379 events × 159.4 candidates × 37 columns ≈ **479 GB uncompressed, ~55.7
GB at parquet+zstd** — inside the spec's 40–160 GB estimate, and the reason that
job is the one genuinely cluster-shaped step. This build ran 2,091,944 rows of
it on a laptop.

An earlier version of this section said ~110 GB, from `positives × candidates`.
That over-counts by the basket size; the same error would have made the shipped
159.4-candidate budget look like ~175 GB and out of envelope, when it is 55.7 GB
and inside it.

## Stage 1: is the two-tower worth keeping?

The retrieval model **loses to the naive baselines on its own** — recall@100 of
6.66% against a popularity+repurchase union's 6.97%, on 20,000 held-out
customers and 70,715 true pairs. By week 3's original checkpoint ("beat the
baseline union") it fails.

It is kept anyway, and the reason is that the checkpoint asked the wrong
question. Stage 1 is not a single retriever, it is a **union of candidate
sources**, and a source earns its place by what the union loses without it, per
candidate slot it occupies — not by whether it beats the whole union alone.

Measured on that basis, at the shipped budget of 159.4 candidates per customer:

| source | solo ceiling | reach | marginal | slots | **marginal/slot** |
|---|---|---|---|---|---|
| repurchase | 2.92% | 92.9% | 1.63% | 18.6 | **0.0876%** |
| co-visitation | 4.81% | 74.4% | 2.66% | 34.2 | 0.0779% |
| **two-tower (ANN)** | **4.16%** | 100.0% | 1.58% | 28.5 | **0.0554%** |
| global popularity | 3.17% | 100.0% | 0.96% | 20.0 | 0.0479% |
| category popularity | 2.16% | 92.9% | 1.11% | 30.5 | 0.0362% |
| **union** | **11.93%** | | 159.4 | |

The tower clears the weakest heuristic on marginal-per-slot — 0.0554% against
0.0362% — so it stays. The rule was fixed *before* any of these numbers existed,
which is what stops it being a rationalisation, and the tower has now cleared it
in **three consecutive tables** built on different candidate configurations
(0.0573/0.0382, then 0.0554/0.0362, then 0.0546/0.0363 with a full-scale tower).

**Final, not provisional.** Both planned experiments ran, and the earlier claim
here — that they "can only move the tower up" — was wrong:

* **Mixed negative sampling** is a **null**. `n_uniform` ∈ {16, 64, 256} against
  0: the best rung beat baseline by 0.051 recall points, which is *exactly* the
  measured seed-noise spread, with no monotone response.
* **Full-scale training** moved recall@100 by **+0.038**, also inside the noise
  floor — and it made the *ceiling slightly worse* (11.93% → 11.84%), because a
  larger share of what the bigger tower retrieves is already covered by
  co-visitation and popularity. A source can improve in isolation and get worse
  at the margin; this one did.

**What ships is the laptop-trained tower** (`r2_recency`, d=64, 2.9M positives),
not the GPU one. Note the consequence: `n_uniform=16` — R.3's nominal best — is
**not in the shipped path**, because the tower that ships was trained with
`n_uniform=0`. The full-scale run cost a GPU allocation and bought nothing
measurable at the metric stage 1 is hired for.

### What actually fixed retrieval, and what did not

| change | recall@100 | vs. noise floor (0.051) |
|---|---|---|
| starting point | 5.53% | — |
| feature spine repair | 5.79% | +0.26, 5x |
| **recency-weighted positives** | **6.66%** | **+0.87, 17x** |
| article volume features | 5.32% | **−0.46, a leak** |

Weighting recent purchases more heavily closed 74% of the gap with one
hyperparameter. Feeding the tower article popularity features — the intuitive
fix, and the one the recovery plan recommended — made it **worse**, and the loss
curve says why: those features lower training loss at every epoch while lowering
recall at every epoch. Each article in a training batch carries volume as of its
own event day, so the model can identify the right answer by matching
day-stamps rather than learning taste; at scoring time every article shares one
day, the shortcut evaporates, and what was learned on it is worthless.

That is the second time in this project that the lower training loss belonged to
the worse model — the first was omitting the sampled-softmax logQ correction,
worth 17x on recall@500. Both were caught by measuring held-out retrieval rather
than watching the objective.

### Known limits of this result

- The candidate ceiling is **11.93%**, against the 12% this stage targeted —
  **missed by 0.070 points**, and shipped anyway. The measured case for not
  buying it back: the last increment of co-visitation lookback returned
  0.0469% per added slot, the worst of any spend in the build, and extending it
  again projects to *touch* 12% rather than clear it. Removing the least
  efficient source was measured and made things worse (9.58%).
- **Co-visitation reaches 74.4% of customers**, so a quarter of the cohort gets
  nothing from the strongest per-slot source. Reach, not depth, is what raised
  this ceiling: 45.9% → 66.7% → 74.4% as the lookback widened, and each step
  bought less than the last.
- The candidate budget is **159.4 per customer**, ~1.6× the ~100 that the
  sizing above was designed around. That was accepted with the cost stated
  (~55.7 GB compressed), not waved through. A depth trim on the weakest source
  was considered and **not taken — because storage is not scarce here (0.5% of
  the available quota), not because a trim was measured cheap.** It was never
  priced.
- The tower is reduced scale by choice, not by limitation: 300,000 of 1.37M
  customers, 2.9M of ~28M positives, d=64. The full-scale alternative was
  trained and measured, and is not the one that ships.
