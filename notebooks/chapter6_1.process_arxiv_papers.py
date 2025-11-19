# Databricks notebook source
# MAGIC %pip install arxiv

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import arxiv
import time
from pyspark.sql import SparkSession
from pyspark.sql import types as T, functions as F
import os
from loguru import logger

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

catalog_name = "mlops_dev"
schema_name = "arxiv"
volume_name = "arxiv_papers"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog_name}.{schema_name}.{volume_name}")

# COMMAND ----------

# search for papers in arxiv in the cs.AI category
# https://arxiv.org/category_taxonomy
# interested in the AI category within computer science:
# cs.AI - Artificial Intelligence

# The expected format for date is [YYYYMMDDTTTT+TO+YYYYMMDDTTTT]
# were the TTTT is provided in 24 hour time to the minute, in GMT.

client = arxiv.Client()

if spark.catalog.tableExists(f"{catalog_name}.{schema_name}.arxiv_papers"):
    start = str(spark.sql(f"""
        SELECT max(processed)
        FROM {catalog_name}.{schema_name}.arxiv_papers
        """).collect()[0][0])
else:
    start = time.strftime("%Y%m%d%H%M", (time.gmtime(time.time() - 24 * 3600 * 3)))

end = time.strftime("%Y%m%d%H%M", time.gmtime())

search = arxiv.Search(
    query=f"cat:cs.AI AND submittedDate:[{start} TO {end}]"
)

papers = client.results(search)

# COMMAND ----------

# create delta table with information about papers,
# including the location of the PDF file in volume storage

records = []

for paper in papers:
    paper_id = paper.get_short_id()

    # download PDF
    pdf_dir = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}/{end}"
    os.makedirs(pdf_dir, exist_ok=True)
    retries = 3
    for attempt in range(retries):
        try:
            paper.download_pdf(dirpath=pdf_dir, filename=f"{paper_id}.pdf")
            # collect metadata (keep datetime intact)
            records.append({
                "paper_id": paper_id,
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "summary": paper.summary,
                "pdf_url": paper.pdf_url,
                "published": int(paper.published.strftime("%Y%m%d%H%M")),
                "processed": int(f"{end}"),
                "volume_path": f"{pdf_dir}/{paper_id}.pdf"
            })
            break
        except Exception as e:
            time.sleep(1)
            if attempt == retries - 1:
                logger.warning(f"Paper {paper_id} was not succesfully processed.")
                pass
    # otherwise hit api limits
    time.sleep(1)

# COMMAND ----------

if len(records) > 0:

    schema = T.StructType([
        T.StructField("paper_id", T.StringType(), False),
        T.StructField("title", T.StringType(), True),
        T.StructField("authors", T.ArrayType(T.StringType()), True),
        T.StructField("summary", T.StringType(), True),
        T.StructField("pdf_url", T.StringType(), True),
        T.StructField("published", T.LongType(), True),
        T.StructField("processed", T.LongType(), True),
        T.StructField("volume_path", T.StringType(), True),
    ])

    # create DataFrame
    df = spark.createDataFrame(records, schema=schema) \
        .withColumn("ingest_ts", F.current_timestamp())

    # write to UC
    df.write \
    .format("delta") \
    .mode("append") \
    .saveAsTable(f"{catalog_name}.{schema_name}.arxiv_papers")


    spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {catalog_name}.{schema_name}.ai_parsed_docs (
    path STRING,
    parsed_content STRING,
    processed LONG)
    """)

    spark.sql(f"""INSERT INTO {catalog_name}.{schema_name}.ai_parsed_docs
    SELECT
    path,
    ai_parse_document(content) AS parsed_content,
    {end} AS processed
    FROM READ_FILES(
    "{pdf_dir}",
    format => 'binaryFile')
    """)

# COMMAND ----------

import json
import re
from pyspark.sql.functions import udf, explode, col, concat_ws
from pyspark.sql.types import ArrayType, StructType, StructField, StringType


df = spark.table(f"{catalog_name}.{schema_name}.ai_parsed_docs").where(f"processed = {end}")

# Define schema for the extracted chunks
chunk_schema = ArrayType(StructType([
    StructField("chunk_id", StringType(), True),
    StructField("content", StringType(), True)
]))

# UDF to extract chunks from parsed_content JSON
def extract_chunks(parsed_content_json):
    if not parsed_content_json:
        return []

    try:
        parsed_dict = json.loads(parsed_content_json)
        chunks = []

        for element in parsed_dict.get("document", {}).get("elements", []):
            if element.get("type") == "text":
                chunk_id = element.get("id", "")
                content = element.get("content", "")
                chunks.append((chunk_id, content))
        return chunks
    except Exception:
        # Return empty list if JSON parsing fails
        return []

extract_chunks_udf = udf(extract_chunks, chunk_schema)

# UDF to extract paper ID from path (last element before .pdf)
def extract_paper_id(path):
    if not path:
        return ""

    # Remove .pdf extension and get the last part of the path
    return path.replace(".pdf", "").split("/")[-1]

extract_paper_id_udf = udf(extract_paper_id, StringType())

# UDF to clean chunk text
def clean_chunk(text: str) -> str:
    if not text:
        return ""

    # trim ends
    t = text.strip()

    # fix hyphenation across line breaks:
    # "docu-\nments" => "documents"
    t = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', t)

    # collapse internal newlines into spaces
    t = re.sub(r'\s*\n\s*', ' ', t)

    # collapse repeated whitespace
    t = re.sub(r'\s+', ' ', t)

    return t.strip()

clean_chunk_udf = udf(clean_chunk, StringType())

# COMMAND ----------

# Create the transformed table
chunks_df = (df
    .withColumn("paper_id", extract_paper_id_udf(col("path")))
    .withColumn("chunks", extract_chunks_udf(col("parsed_content")))
    .withColumn("chunk", explode(col("chunks")))
    .select(
        col("paper_id"),
        col("chunk.chunk_id").alias("chunk_id"),
        clean_chunk_udf(col("chunk.content")).alias("text"),
        concat_ws("_", col("paper_id"), col("chunk.chunk_id")).alias("id")
    )
)

# COMMAND ----------
# Write to table
table_name = f"{catalog_name}.{schema_name}.arxiv_chunks"
chunks_df.write.mode("append").saveAsTable(table_name)

# COMMAND ----------
spark.sql(f"""ALTER TABLE {catalog_name}.{schema_name}.arxiv_chunks
           SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
          """)
