{{ mart_config() }}

-- Current-state snapshot, NOT point-in-time: articles.csv has no history and no
-- valid-from/valid-to, so a 2018 event joins to 2020 attribute values. Stated in
-- the README's limitations; a Type-2 SCD is the standard fix and this dataset
-- cannot support one.
select
    article_id,
    product_name,
    product_type_no,
    product_type_name,
    product_group_name,
    colour_group_code,
    colour_group_name,
    department_no,
    department_name,
    index_code,
    index_name,
    index_group_no,
    index_group_name,
    section_no,
    section_name,
    garment_group_no,
    garment_group_name
from {{ ref('stg_articles') }}
