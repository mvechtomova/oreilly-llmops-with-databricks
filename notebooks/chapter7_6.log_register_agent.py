# Databricks notebook source
# https://docs.databricks.com/aws/en/generative-ai/mcp/managed-mcp
# https://docs.databricks.com/aws/en/mlflow3/genai/tracing/prod-tracing
# https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent

# %pip install arxiv_curator-0.1.1-py3-none-any.whl

# COMMAND ----------
import mlflow
from arxiv_agent import agent
from mlflow.models.resources import (
    DatabricksGenieSpace,
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
)

from arxiv_curator.config import ProjectConfig

# COMMAND ----------
# Test the agent
cfg = ProjectConfig.from_yaml("../project_config.yml")
catalog = cfg.catalog
schema = cfg.schema
genie_space_id = cfg.genie_space_id
llm_endpoint = cfg.llm_endpoint
system_prompt = cfg.system_prompt
model_name = "arxiv_agent"

test_request = {
    "input": [
        {"role": "user",
         "content": "What are recent papers about LLMs and reasoning?"}
    ]
}

# COMMAND ----------
# Run the agent
# result = agent.predict(test_request)
# print(result.model_dump(exclude_none=True))

# COMMAND ----------
# Run the agent with streaming
for chunk in agent.predict_stream(test_request):
    print(chunk.model_dump(exclude_none=True))

# COMMAND ----------
# Log and register the agent with MLflow

git_sha = "abc"  # Replace with actual git sha in production
run_id = "unset"  # Replace with actual run id in production


resources = [
    DatabricksServingEndpoint(endpoint_name=llm_endpoint),
    DatabricksGenieSpace(genie_space_id=genie_space_id),
    DatabricksVectorSearchIndex(index_name=f"{catalog}.{schema}.arxiv_index"),
]

test_request = {
    "input": [
        {"role": "user", "content": "What are recent papers about LLMs and reasoning?"}
    ]
}

model_config = {
        "catalog": catalog,
        "schema": schema,
        "genie_space_id": genie_space_id,
        "system_prompt": system_prompt,
        "llm_endpoint": llm_endpoint,
    }

mlflow.set_experiment("/Shared/genai-arxiv-agent")
with mlflow.start_run(
    run_name="arxiv-mcp-agent",
    tags={"git_sha": git_sha, "run_id": run_id}
) as run:
    model_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="arxiv_agent.py",
        resources=resources,
        input_example=test_request,
        model_config=model_config
    )

# COMMAND ----------
# Register the model to Unity Catalog


registered_model = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=f"{catalog}.{schema}.{model_name}",
    env_pack="databricks_model_serving"
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
    tags={"endpointSource": "docs"},
    deploy_feedback_model=False,
)
