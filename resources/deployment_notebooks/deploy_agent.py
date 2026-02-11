# Databricks notebook source
from databricks.sdk.runtime import dbutils

from arxiv_curator.config import ProjectConfig
from arxiv_curator.serving import serve_model
from arxiv_curator.utils.common import get_widget

# Get model_version from previous task or widget
model_version = dbutils.jobs.taskValues.get(
    taskKey="log_register_agent",
    key="model_version",
)
git_sha = get_widget("git_sha", "local")
env = get_widget("env", "dev")

cfg = ProjectConfig.from_yaml("../../project_config.yml",
                              env=env)

model_name = f"{cfg.catalog}.{cfg.schema}.arxiv_agent"
endpoint_name = "arxiv-agent-endpoint"

serve_model(
    entity_name=model_name,
    entity_version=model_version,
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
