-- Rename and cast only. No business logic: the grain is resolved in
-- fact_transaction, one layer up.
select
    t_dat                       as transaction_date,
    customer_id,
    article_id,
    cast(price as double)       as price,
    cast(sales_channel_id as int) as sales_channel_id,
    _ingested_at
from {{ raw_source('transactions') }}
