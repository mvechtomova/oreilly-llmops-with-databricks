# Databricks notebook source
# https://docs.databricks.com/aws/en/generative-ai/mcp/managed-mcp
# https://docs.databricks.com/aws/en/mlflow3/genai/tracing/prod-tracing

%pip install arxiv_curator-0.1.1-py3-none-any.whl

# COMMAND ----------
from agent import agent, llm_endpoint

import mlflow
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.models.resources import DatabricksServingEndpoint, DatabricksGenieSpace, DatabricksVectorSearchIndex
# COMMAND ----------
# Test the agent

catalog = "mlops_dev"
schema = "arxiv"
model_name = "arxiv_agent"

test_request =  {"input": [{"role": "user",
                "content": "What are recent papers about LLMs and reasoning?"}]}


# Run the agent
# result = agent.predict(test_request)
# print(result.model_dump(exclude_none=True))

# Run the agent with streaming
for chunk in agent.predict_stream(test_request):
    print(chunk.model_dump(exclude_none=True))

# COMMAND ----------
# Log and register the agent with MLflow


resources = [DatabricksServingEndpoint(endpoint_name=llm_endpoint),
             DatabricksGenieSpace(genie_space_id="01f0c3f882e41bb9be96b4ddf295d2a4"),
             DatabricksVectorSearchIndex(index_name=f"{catalog}.{schema}.arxiv_index")]

code_paths = ["arxiv_curator-0.1.1-py3-none-any.whl"]
additional_pip_deps = []
for package in code_paths:
    whl_name = package.split("/")[-1]
    additional_pip_deps.append(f"code/{whl_name}")

with mlflow.start_run(run_name="arxiv-mcp-agent",
                      tags={"git_sha": "abc",
                            "run_id": "unset"}) as run:
    model_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        resources=resources,
        input_example=test_request,
        conda_env=_mlflow_conda_env(additional_pip_deps=additional_pip_deps),
        code_paths=code_paths,
    )

# COMMAND ----------
# Register the model to Unity Catalog


registered_model = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=f"{catalog}.{schema}.{model_name}",
)


# COMMAND ----------

mlflow.models.predict(
    model_uri=model_info.model_uri,
    input_data=test_request,
)
# COMMAND ----------
from databricks import agents

agents.deploy(
    f"{catalog}.{schema}.{model_name}",
    registered_model.version,
    tags = {"endpointSource": "docs"},
    deploy_feedback_model=False)
