from mlflow import MlflowClient

from arxiv_curator.config import ProjectConfig
from arxiv_curator.serving import serve_model

cfg = ProjectConfig.from_yaml("../../project_config.yml")

model_name = f"{cfg.catalog}.{cfg.schema}.arxiv_agent"
endpoint_name = "arxiv-agent-endpoint"

client = MlflowClient()
model_version = client.get_model_version_by_alias(
    alias="latest-model",
    name=model_name)

# COMMAND ----------
git_sha = "local"
serve_model(
    entity_name=model_name,
    entity_version=model_version.version,
    tags={"key": "project_name", "value": "arxiv_curator"},
    endpoint_name=endpoint_name,
    catalog_name=cfg.catalog,
    schema_name=cfg.schema,
    table_name_prefix="arxiv_agent_monitoring",
    environment_vars={
        "GIT_SHA": git_sha,
        "MODEL_VERSION": model_version,
        "MODEL_SERVING_ENDPOINT_NAME": endpoint_name,
    },
)