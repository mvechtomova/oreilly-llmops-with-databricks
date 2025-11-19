import argparse

import mlflow
from agent import llm_endpoint
from mlflow import MlflowClient
from mlflow.models.resources import DatabricksServingEndpoint
from mlflow.utils.environment import _mlflow_conda_env

from arxiv_curator.config import ProjectConfig

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

parser.add_argument(
    "--git_sha",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--run_id",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--deployment_job_id",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"
code_paths = [f"{root_path}/artifacts/.internal/arxiv_curator-0.1.1-py3-none-any.whl"]

project_config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)


test_request =  {"input": [{"role": "user",
                "content": "What are recent papers about LLMs and reasoning?"}]}


resources = [DatabricksServingEndpoint(endpoint_name=llm_endpoint)]

additional_pip_deps = []
for package in code_paths:
    whl_name = package.split("/")[-1]
    additional_pip_deps.append(f"code/{whl_name}")

tags={"git_sha": args.git_sha, "run_id": args.run_id}

with mlflow.start_run(run_name="arxiv-mcp-agent",
                      tags=tags) as run:
    model_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        resources=resources,
        input_example=test_request,
        conda_env=_mlflow_conda_env(additional_pip_deps=additional_pip_deps),
        code_paths=code_paths,
    )

# Register the model to Unity Catalog
catalog = project_config.catalog_name
schema = project_config
model_name = f"{catalog}.{schema}.arxiv_agent"

registered_model = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=model_name,
    tags=tags,
)

client = MlflowClient()
client.set_registered_model_alias(
    name=model_name,
    alias="latest-model",
    version=registered_model.version,
)

client.update_registered_model(model_name=model_name,
                            deployment_job_id=args.deployment_job_id)
