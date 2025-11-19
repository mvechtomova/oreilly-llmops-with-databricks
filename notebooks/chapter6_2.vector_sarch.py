# Databricks notebook source
from pyspark.sql import SparkSession
from loguru import logger

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

catalog_name = "mlops_dev"
schema_name = "arxiv"


# COMMAND ----------
# vector search
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()
vector_search_endpoint_name = "vector-search-arxiv-endpoint"

vsc.create_endpoint(
    name=vector_search_endpoint_name,
    endpoint_type="STANDARD"
)

# COMMAND ----------
vs_index_fullname = f"{catalog_name}.{schema_name}.arxiv_index"

embedding_model_endpoint = "databricks-gte-large-en"

index = vsc.create_delta_sync_index(
  endpoint_name=vector_search_endpoint_name,
  source_table_name=f"{catalog_name}.{schema_name}.arxiv_chunks",
  index_name=vs_index_fullname,
  pipeline_type='TRIGGERED',
  primary_key="id",
  embedding_source_column="text",
  embedding_model_endpoint_name=embedding_model_endpoint
)
# COMMAND ----------
index.describe()

# COMMAND ----------
results = index.similarity_search(
  query_text="AI development is fast",
  columns=["text", "id"],
  num_results=5)
