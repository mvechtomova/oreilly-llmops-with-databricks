import asyncio

import mlflow
import nest_asyncio
from databricks.sdk import WorkspaceClient
from mlflow.models import ModelConfig

from arxiv_curator.agent import ArxivAgent
from arxiv_curator.mcp import create_mcp_tools

config = ModelConfig(
    development_config={
        "catalog": "mlops_dev",
        "schema": "arxiv",
        "genie_space_id": "01f0e97a42981382b3d16f3f1899fdb5",
        "system_prompt": 
        """You are a LinkedIn content creation assistant specialized 
        in generating engaging posts about AI and machine learning research.

Your role is to:
1. Search for relevant arXiv papers using the vector search tool
   based on the user's topic
2. Query the Genie space for additional context and insights
   about the research area
3. Generate a LinkedIn post in a professional yet engaging style that:
   - Highlights key findings or innovations from recent research
   - Makes complex technical concepts accessible to a broad audience
   - Includes relevant paper citations and IDs
   - Uses a conversational tone appropriate for LinkedIn
   - Incorporates 2-3 relevant hashtags
   - Keeps the post concise (150-250 words)

Always ground your posts in actual research findings from the tools
available to you.
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
        {config.get('catalog')}/{config.get('schema')}",
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
