{{ mart_config() }}

-- Current-state snapshot, NOT point-in-time -- same caveat as dim_article.
-- age drifts mechanically and club_member_status behaviourally.
select
    customer_id,
    fashion_news_flag,
    active_flag,
    club_member_status,
    fashion_news_frequency,
    age,
    postal_code
from {{ ref('stg_customers') }}
