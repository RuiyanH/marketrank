-- Rename and cast only. FN and Active arrive as sparse 1.0/null doubles, which
-- is a delivery artifact rather than three-valued logic: null means "not set".
-- age keeps its nulls -- that IS information, and the seeds carry one.
select
    cast(customer_id as string)                     as customer_id,
    coalesce(FN, 0) = 1                             as fashion_news_flag,
    coalesce(Active, 0) = 1                         as active_flag,
    club_member_status,
    fashion_news_frequency,
    cast(age as int)                                as age,
    postal_code
from {{ raw_source('customers') }}
