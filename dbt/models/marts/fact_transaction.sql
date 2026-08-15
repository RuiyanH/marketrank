{{ mart_config() }}

-- Grain: ONE BASKET LINE -- (customer, article, day, channel) -- with a qty
-- count. See step 1.4. The key is a function of the business grain, not of row
-- order, which is what keeps it stable across re-runs.
--
-- price is NOT in the key: 0.78% of basket lines saw more than one price
-- (a mid-day markdown), and that is not a second basket line. revenue is
-- sum(price), which is exact and equals qty * price_mean even in those groups.
select
    customer_id,
    article_id,
    transaction_date,
    sales_channel_id,
    count(*)                as qty,
    sum(price)              as revenue,
    sum(price) / count(*)   as price_mean,
    min(price)              as price_min,
    max(price)              as price_max,
    max(_ingested_at)       as _ingested_at
from {{ ref('stg_transactions') }}
group by 1, 2, 3, 4
