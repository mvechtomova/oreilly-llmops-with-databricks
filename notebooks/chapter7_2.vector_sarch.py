# Databricks notebook source
from pyspark.sql import SparkSession
from arxiv_curator.config import ProjectConfig

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

project_config = ProjectConfig.from_yaml("../project_config.yml")
catalog_name = project_config.catalog_name
schema_name = project_config.schema_name


# COMMAND ----------
# vector search
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()
vector_search_endpoint_name = "vector-search-arxiv-endpoint"
endpoint_exists = any(
    item.name == vector_search_endpoint_name for item in vsc.list_endpoints()
)
if not endpoint_exists:
    vsc.create_endpoint(name=vector_search_endpoint_name,
                        endpoint_type="STANDARD")

# COMMAND ----------
vs_index_fullname = f"{catalog_name}.{schema_name}.arxiv_index"

embedding_model_endpoint = "databricks-gte-large-en"



vs_index_fullname = f"{catalog_name}.{schema_name}.arxiv_index"
embedding_model_endpoint = "databricks-gte-large-en"

index_exists = any(
    item.name == vector_search_endpoint_name for item in vsc.list_indexes(vector_search_endpoint_name)
)

if not index_exists:
    index = vsc.create_delta_sync_index(
        endpoint_name=vector_search_endpoint_name,
        source_table_name=f"{catalog_name}.{schema_name}.arxiv_chunks",
        index_name=vs_index_fullname,
        pipeline_type="TRIGGERED",
        primary_key="id",
        embedding_source_column="text",
        embedding_model_endpoint_name=embedding_model_endpoint,
    )
else:
    index = vsc.get_index(index_name=vs_index_fullname)
    index.sync()


# COMMAND ----------
from databricks.vector_search.reranker import DatabricksReranker

results = index.similarity_search(
    query_text="Chunking strategies for document processing",
    columns=["text", "id"],
    filters={'year': "2026"},
    num_results=5,
    query_type = "hybrid",
    reranker=DatabricksReranker(columns_to_rerank=["text", "title", "summary"])
    )
