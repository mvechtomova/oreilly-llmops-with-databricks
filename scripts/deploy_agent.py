from databricks import agents
from agent import llm_endpoint

import mlflow
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.models.resources import DatabricksServingEndpoint
from arxiv_curator.config import ProjectConfig
import argparse
from mlflow import MlflowClient

model_version = dbutils.jobs.taskValues.get(taskKey="approve", key="model_version")

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

catalog= project_config.catalog_name
schema= project_config.schema_name

agents.deploy(
    f"{catalog}.{schema}.arxiv_agent",
    model_version,
    tags = {"project": "arxiv-curator"},
    deploy_feedback_model=False)
