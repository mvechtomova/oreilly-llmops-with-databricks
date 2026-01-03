import json
import os
import re
import time

import arxiv
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import types as T
from pyspark.sql.functions import col, concat_ws, current_timestamp, explode, udf
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

from arxiv_curator.config import ProjectConfig


class DataProcessor:
    """
    DataProcessor handles the complete workflow of:
    - Downloading papers from arXiv
    - Storing paper metadata
    - Parsing PDFs with ai_parse_document
    - Extracting and cleaning text chunks
    - Saving chunks to Delta tables
    """

    def __init__(self, spark: SparkSession, config: ProjectConfig) -> None:
        """
        Initialize DataProcessor with Spark session and configuration.

        Args:
            spark: SparkSession instance
            config: ProjectConfig object with table configurations
        """
        self.spark = spark
        self.config = config
        self.catalog_name = config.catalog_name
        self.schema_name = config.schema_name
        self.volume_name = config.volume_name

        # Define schema for the extracted chunks
        self.chunk_schema = ArrayType(
            StructType(
                [
                    StructField("chunk_id", StringType(), True),
                    StructField("content", StringType(), True),
                ]
            )
        )

        # Register UDFs
        self.extract_chunks_udf = udf(self._extract_chunks, self.chunk_schema)
        self.extract_paper_id_udf = udf(self._extract_paper_id, StringType())
        self.clean_chunk_udf = udf(self._clean_chunk, StringType())

        self.end = time.strftime("%Y%m%d%H%M", time.gmtime())
        self.pdf_dir = f"/Volumes/{self.catalog_name}/{self.schema_name}/{self.volume_name}/{self.end}"
        os.makedirs(self.pdf_dir, exist_ok=True)

    def _get_range_start(self) -> tuple[str, str]:
        """
        Get start time range for arxiv paper search.
        If arxiv_papers table exists, uses max(processed) as start.
        Otherwise, uses 3 days ago as start.

        Returns:
            start string in "YYYYMMDDHHMM" format
        """
        arxiv_papers_table = self.config.get_full_table_name("arxiv_papers_table")

        if self.spark.catalog.tableExists(arxiv_papers_table):
            result = self.spark.sql(f"""
                SELECT max(processed)
                FROM {arxiv_papers_table}
            """).collect()
            start = str(result[0][0])
            logger.info(f"Found existing arxiv_papers table. Starting from: {start}")
        else:
            start = time.strftime("%Y%m%d%H%M", time.gmtime(time.time() - 24 * 3600 * 3))
            logger.info(
                f"No existing arxiv_papers table. Starting from 3 days ago: {start}"
            )

        return start

    def download_and_store_papers(self) -> tuple[list[dict], str] | tuple[None, None]:
        """
        Download papers from arxiv and store metadata in arxiv_papers table.

        Returns:
            Tuple of (records, pdf_dir) if papers were downloaded, otherwise (None, None)
        """
        start = self._get_range_start()

        # Search for papers in arxiv
        client = arxiv.Client()
        search = arxiv.Search(
            query=f"cat:cs.AI AND submittedDate:[{start} TO {self.end}]"
        )
        papers = client.results(search)

        # Download papers and collect metadata
        records = []

        for paper in papers:
            paper_id = paper.get_short_id()

            retries = 3
            for attempt in range(retries):
                try:
                    paper.download_pdf(dirpath=self.pdf_dir, filename=f"{paper_id}.pdf")
                    # Collect metadata
                    records.append(
                        {
                            "paper_id": paper_id,
                            "title": paper.title,
                            "authors": [author.name for author in paper.authors],
                            "summary": paper.summary,
                            "pdf_url": paper.pdf_url,
                            "published": int(paper.published.strftime("%Y%m%d%H%M")),
                            "processed": int(self.end),
                            "volume_path": f"{self.pdf_dir}/{paper_id}.pdf",
                        }
                    )
                    break
                except Exception:
                    time.sleep(1)
                    if attempt == retries - 1:
                        logger.warning(
                            f"Paper {paper_id} was not successfully processed."
                        )
            # Avoid hitting API rate limits
            time.sleep(1)

        # Only process if we have records
        if len(records) == 0:
            logger.info("No new papers found.")
            return None, None

        logger.info(f"Downloaded {len(records)} papers to {self.pdf_dir}")

        # Create DataFrame and save to arxiv_papers table
        schema = T.StructType(
            [
                T.StructField("paper_id", T.StringType(), False),
                T.StructField("title", T.StringType(), True),
                T.StructField("authors", T.ArrayType(T.StringType()), True),
                T.StructField("summary", T.StringType(), True),
                T.StructField("pdf_url", T.StringType(), True),
                T.StructField("published", T.LongType(), True),
                T.StructField("processed", T.LongType(), True),
                T.StructField("volume_path", T.StringType(), True),
            ]
        )

        df = self.spark.createDataFrame(records, schema=schema).withColumn(
            "ingest_ts", current_timestamp()
        )

        arxiv_papers_table = self.config.get_full_table_name("arxiv_papers_table")
        df.write.format("delta").mode("append").saveAsTable(arxiv_papers_table)
        logger.info(f"Saved {len(records)} paper records to {arxiv_papers_table}")

        return records

    def parse_pdfs_with_ai(self) -> None:
        """
        Parse PDFs using ai_parse_document and store in ai_parsed_docs table.

        """
        ai_parsed_docs_table = self.config.get_full_table_name("ai_parsed_docs_table")

        # Create table if it doesn't exist
        self.spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {ai_parsed_docs_table} (
                path STRING,
                parsed_content STRING,
                processed LONG
            )
        """)

        # Parse PDFs and insert into table
        self.spark.sql(f"""
            INSERT INTO {ai_parsed_docs_table}
            SELECT
                path,
                ai_parse_document(content) AS parsed_content,
                {self.end} AS processed
            FROM READ_FILES(
                "{self.pdf_dir}",
                format => 'binaryFile'
            )
        """)

        logger.info(
            f"Parsed PDFs from {self.pdf_dir} and saved to {ai_parsed_docs_table}"
        )

    @staticmethod
    def _extract_chunks(parsed_content_json: str) -> list[tuple[str, str]]:
        """
        Extract chunks from parsed_content JSON.

        Args:
            parsed_content_json: JSON string containing parsed document structure

        Returns:
            List of tuples containing (chunk_id, content)
        """
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

    @staticmethod
    def _extract_paper_id(path: str) -> str:
        """
        Extract paper ID from file path.

        Args:
            path: File path (e.g., "/path/to/paper_id.pdf")

        Returns:
            Paper ID extracted from the path
        """
        if not path:
            return ""

        # Remove .pdf extension and get the last part of the path
        return path.replace(".pdf", "").split("/")[-1]

    @staticmethod
    def _clean_chunk(text: str) -> str:
        """
        Clean and normalize chunk text by:
        - Trimming whitespace
        - Fixing hyphenation across line breaks
        - Collapsing internal newlines into spaces
        - Collapsing repeated whitespace

        Args:
            text: Raw text content

        Returns:
            Cleaned text content
        """
        if not text:
            return ""

        # Trim ends
        t = text.strip()

        # Fix hyphenation across line breaks:
        # "docu-\nments" => "documents"
        t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)

        # Collapse internal newlines into spaces
        t = re.sub(r"\s*\n\s*", " ", t)

        # Collapse repeated whitespace
        t = re.sub(r"\s+", " ", t)

        return t.strip()

    def process_chunks(self) -> None:
        """
        Process parsed documents to extract and clean chunks.
        Reads from ai_parsed_docs table and saves to arxiv_chunks table.
        """
        ai_parsed_docs_table = self.config.get_full_table_name("ai_parsed_docs_table")
        logger.info(
            f"Processing parsed documents from {ai_parsed_docs_table} for end date {self.end}"
        )

        df = self.spark.table(ai_parsed_docs_table).where(f"processed = {self.end}")

        # Create the transformed dataframe
        chunks_df = (
            df.withColumn("paper_id", self.extract_paper_id_udf(col("path")))
            .withColumn("chunks", self.extract_chunks_udf(col("parsed_content")))
            .withColumn("chunk", explode(col("chunks")))
            .select(
                col("paper_id"),
                col("chunk.chunk_id").alias("chunk_id"),
                self.clean_chunk_udf(col("chunk.content")).alias("text"),
                concat_ws("_", col("paper_id"), col("chunk.chunk_id")).alias("id"),
            )
        )

        # Write to table
        arxiv_chunks_table = self.config.get_full_table_name("arxiv_chunks_table")
        chunks_df.write.mode("append").saveAsTable(arxiv_chunks_table)
        logger.info(f"Saved chunks to {arxiv_chunks_table}")

        # Enable Change Data Feed
        self.spark.sql(f"""
            ALTER TABLE {arxiv_chunks_table}
            SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)
        logger.info(f"Change Data Feed enabled for {arxiv_chunks_table}")

    def process_and_save(self) -> None:
        """
        Complete workflow: download papers, parse PDFs, and process chunks.
        Matches the logic from chapter6_1 notebook.
        """
        # Step 1: Download papers and store metadata
        records = self.download_and_store_papers()

        # Only continue if we have new papers
        if records is None or len(records) == 0:
            logger.info("No new papers to process. Exiting.")
            return

        # Step 2: Parse PDFs with ai_parse_document
        self.parse_pdfs_with_ai()

        # Step 3: Process chunks
        self.process_chunks()
        logger.info("Processing complete!")
