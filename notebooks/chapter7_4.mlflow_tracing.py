# Databricks notebook source

# To see detailed optimization output during alignment, enable DEBUG logging:
# import logging
# logging.getLogger("mlflow.genai.judges.optimizers.simba").setLevel(logging.DEBUG)
import asyncio

import mlflow
import nest_asyncio
from databricks.sdk import WorkspaceClient

from arxiv_curator.agent import ArxivAgent
from arxiv_curator.mcp import create_mcp_tools
from arxiv_curator.config import ProjectConfig


cfg = ProjectConfig.from_yaml("../project_config.yml")

# COMMAND ----------
# Set the agent
nest_asyncio.apply()

w = WorkspaceClient()
host = w.config.host

MANAGED_MCP_SERVER_URLS = [
    f"{host}/api/2.0/mcp/genie/{cfg.genie_space_id}",
    f"{host}/api/2.0/mcp/vector-search/{cfg.catalog}/{cfg.schema}",
]

tools = asyncio.run(
    create_mcp_tools(w=w, url_list=MANAGED_MCP_SERVER_URLS,)
)

agent = ArxivAgent(
    llm_endpoint=cfg.llm_endpoint,
    tools=tools,
    system_prompt=cfg.system_prompt
)
mlflow.models.set_model(agent)

# COMMAND ----------
# Test the agent

test_request = {
    "input": [
        {"role": "user",
         "content": "What are recent papers about LLMs and reasoning?"}
    ]
}

result = agent.predict(test_request)
print(result.model_dump(exclude_none=True))

# COMMAND ----------
# Run the agent with streaming
for chunk in agent.predict_stream(test_request):
    print(chunk.model_dump(exclude_none=True))

