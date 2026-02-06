# Databricks notebook source

# COMMAND ----------
# Test the agent
import random
from datetime import datetime

from arxiv_agent import agent

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
session_id = f"s-{timestamp}-{random.randint(100000, 999999)}"
request_id = f"req-{timestamp}-{random.randint(100000, 999999)}"

test_request = {
    "input": [
        {"role": "user",
         "content": "What are recent papers about LLMs and reasoning?"}
    ],
    "custom_inputs": {
        "session_id": session_id,
        "request_id": request_id,
    },
}

result = agent.predict(test_request)
print(result.model_dump(exclude_none=True))

# Run the agent with streaming
for chunk in agent.predict_stream(test_request):
    print(chunk.model_dump(exclude_none=True))

# COMMAND ----------
# Log and register the agent with MLflow
from datetime import datetime

import mlflow
from mlflow.models.resources import (
    DatabricksGenieSpace,
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
    DatabricksTable,
    DatabricksSQLWarehouse
)

from arxiv_curator.config import ProjectConfig


cfg = ProjectConfig.from_yaml("../project_config.yml")

model_name = f"{cfg.catalog}.{cfg.schema}.arxiv_agent"

resources = [
    DatabricksServingEndpoint(endpoint_name=cfg.llm_endpoint),
    DatabricksGenieSpace(genie_space_id=cfg.genie_space_id),
    DatabricksVectorSearchIndex(
        index_name=f"{cfg.catalog}.{cfg.schema}.arxiv_index"),
    DatabricksTable(table_name=f"{cfg.catalog}.{cfg.schema}.arxiv_papers"),
    DatabricksSQLWarehouse(warehouse_id=cfg.warehouse_id),
    DatabricksServingEndpoint(endpoint_name="databricks-bge-large-en"),
]

model_config = {
        "catalog": cfg.catalog,
        "schema": cfg.schema,
        "genie_space_id": cfg.genie_space_id,
        "system_prompt": cfg.system_prompt,
        "llm_endpoint": cfg.llm_endpoint,
    }

git_sha = "abc"
run_id = "unset"

mlflow.set_experiment("/Shared/genai-arxiv-agent")
ts = ts = datetime.now().strftime('%Y-%m-%d')
with mlflow.start_run(
    run_name=f"arxiv-agent-{ts}",
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
    name=model_name,
    env_pack="databricks_model_serving"
)


# COMMAND ----------
# Deploy the model to a serving endpoint
from arxiv_curator.serving import serve_model, call_endpoint

endpoint_name = "arxiv-agent-endpoint"

serve_model(
    entity_name=model_name,
    entity_version=str(registered_model.version),
    tags={"key": "project_name", "value": "arxiv_curator"},
    endpoint_name=endpoint_name,
    catalog_name=cfg.catalog,
    schema_name=cfg.schema,
    table_name_prefix="arxiv_agent_monitoring",
    environment_vars={
        "GIT_SHA": git_sha,
        "MODEL_VERSION": str(registered_model.version),
        "MODEL_SERVING_ENDPOINT_NAME": endpoint_name,
    },
)

# COMMAND ----------
# Call the endpoint with trace metadata
import random
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
session_id = f"s-{timestamp}-{random.randint(100000, 999999)}"
request_id = f"req-{timestamp}-{random.randint(100000, 999999)}"

response = call_endpoint(
    endpoint_name=endpoint_name,
    messages=[
        {"role": "user", "content": "What are recent papers about LLMs and reasoning?"}
    ],
    custom_inputs={
        "session_id": session_id,
        "request_id": request_id,
    },
)
print(response)
