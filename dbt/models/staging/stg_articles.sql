-- Rename and cast only. article_id stays a string: it is zero-padded and an
-- integer cast silently breaks every join to transactions.
select
    cast(article_id as string)          as article_id,
    prod_name                           as product_name,
    cast(product_type_no as int)        as product_type_no,
    product_type_name,
    product_group_name,
    cast(colour_group_code as int)      as colour_group_code,
    colour_group_name,
    cast(department_no as int)          as department_no,
    department_name,
    index_code,
    index_name,
    cast(index_group_no as int)         as index_group_no,
    index_group_name,
    cast(section_no as int)             as section_no,
    section_name,
    cast(garment_group_no as int)       as garment_group_no,
    garment_group_name
from {{ raw_source('articles') }}
