import asyncio

import mlflow
import nest_asyncio
from databricks.sdk import WorkspaceClient
from mlflow.models import ModelConfig

from arxiv_curator.agent import ArxivAgent
from arxiv_curator.mcp import create_mcp_tools

config = ModelConfig(
    development_config={
        "catalog_name": "mlops_dev",
        "schema_name": "arxiv",
        "genie_space_id": "01f0e97a42981382b3d16f3f1899fdb5",
        "system_prompt": """You are a helpful AI assistant with access
to tools for searching arXiv papers and querying a Genie space.

When helping users:
- Use the vector search tool to find relevant arXiv papers based on semantic similarity
- Use the Genie query_space tool to answer questions about the data in the space
- Be concise and informative in your responses
- Cite paper IDs when referencing specific papers
""",
        "llm_endpoint": "databricks-gpt-oss-120b",
    }
)

nest_asyncio.apply()
mlflow.set_experiment("/Shared/genai-arxiv-agent")

w = WorkspaceClient()
host = w.config.host

MANAGED_MCP_SERVER_URLS = [
    f"{host}/api/2.0/mcp/genie/{config.get('genie_space_id')}",
    f"{host}/api/2.0/mcp/vector-search/\
        {config.get('catalog_name')}/{config.get('schema_name')}",
]
tools = asyncio.run(
    create_mcp_tools(w=w, url_list=MANAGED_MCP_SERVER_URLS,)
)

agent = ArxivAgent(
    llm_endpoint=config.get("llm_endpoint"),
    tools=tools,
    system_prompt=config.get("system_prompt")
)
mlflow.models.set_model(agent)
