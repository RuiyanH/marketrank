{#
  Schema names as written, not prefixed with the target schema. `marts` means
  `marts`, so the Spark target lands models in `local.marts.*` -- which is what
  the step 1.5 checkpoint queries.
#}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- if custom_schema_name is none -%}
        {{ target.schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}


{#
  One name for "the raw table", two places it can live.

  On Spark it is the real Iceberg table in `local.raw`. On the CI target there
  is no warehouse and no 3.5 GB CSV, so it is the committed seed. The model SQL
  below this line is identical on both -- which is the whole point: CI runs the
  same dimensional model, not a copy of it.
#}
{% macro raw_source(name) -%}
    {%- if target.type == 'duckdb' -%}
        {{ ref('seed_' ~ name) }}
    {%- else -%}
        {{ source('raw', name) }}
    {%- endif -%}
{%- endmacro %}


{#
  Iceberg on Spark, plain table everywhere else. Engine-specific settings go in
  config(), never in the body of a query.
#}
{% macro mart_config() -%}
    {%- if target.type == 'spark' -%}
        {{ config(materialized='table', file_format='iceberg') }}
    {%- else -%}
        {{ config(materialized='table') }}
    {%- endif -%}
{%- endmacro %}
