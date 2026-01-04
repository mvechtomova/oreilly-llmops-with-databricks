import argparse

import yaml
from loguru import logger
from pyspark.sql import SparkSession

from arxiv_curator.config import ProjectConfig
from arxiv_curator.data_processor import DataProcessor
from arxiv_curator.vector_search import VectorSearchManager

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--env",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"

project_config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)

logger.info("Configuration loaded:")
logger.info(yaml.dump(project_config, default_flow_style=False))

spark = SparkSession.builder.getOrCreate()


# Initialize DataProcessor
data_processor = DataProcessor(config=project_config, spark=spark)
data_processor.process_and_save()

# Sync vector search index
logger.info("Syncing vector search index...")
vector_search_manager = VectorSearchManager(config=project_config)
vector_search_manager.sync_index()
logger.info("Vector search index sync complete!")
