-- The fact's grain is composite, so `unique` on a single column cannot express
-- it. Written by hand rather than pulling in dbt_utils for one macro; the SQL is
-- plain enough to run identically on Spark and DuckDB.
--
-- dbt singular tests pass when they return zero rows.
select
    customer_id,
    article_id,
    transaction_date,
    sales_channel_id,
    count(*) as n
from {{ ref('fact_transaction') }}
group by 1, 2, 3, 4
having count(*) > 1
