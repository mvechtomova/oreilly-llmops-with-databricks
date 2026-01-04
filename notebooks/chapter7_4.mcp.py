# Databricks notebook source
from collections.abc import Callable
from pydantic import BaseModel


# COMMAND ----------
class ToolInfo(BaseModel):
    """
    Class representing a tool for the agent.
    - "name" (str): The name of the tool.
    - "spec" (dict): JSON description of the tool (matches OpenAI Responses format)
    - "exec_fn" (Callable): Function that implements the tool logic
    """

    name: str
    spec: dict
    exec_fn: Callable


# COMMAND ----------
from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient


def create_managed_exec_fn(
    server_url: str, tool_name: str, w: WorkspaceClient
) -> Callable:
    def exec_fn(**kwargs):
        client = DatabricksMCPClient(server_url=server_url, workspace_client=w)
        response = client.call_tool(tool_name, kwargs)
        return "".join([c.text for c in response.content])

    return exec_fn


# COMMAND ----------
# from databricks_mcp import DatabricksOAuthClientProvider
# from mcp import ClientSession
# from mcp.client.streamable_http import streamablehttp_client as connect

# https://learn.microsoft.com/en-us/azure/databricks/generative-ai/mcp/managed-mcp-meta-param
# async def _run_custom_async(
#     server_url: str, tool_name: str, ws: WorkspaceClient, **kwargs
# ) -> str:
#     """Executes a tool from a custom MCP server asynchronously using OAuth."""
#     async with connect(server_url, auth=DatabricksOAuthClientProvider(ws)) as (
#         read_stream,
#         write_stream,
#         _,
#     ):
#         async with ClientSession(read_stream, write_stream) as session:
#             await session.initialize()
#             response = await session.call_tool(tool_name, kwargs)
#             return "".join([c.text for c in response.content])



# COMMAND ----------


async def create_mcp_tools(w: WorkspaceClient, url_list: list[str]) -> list[ToolInfo]:
    tools = []
    for server_url in url_list:
        mcp_client = DatabricksMCPClient(server_url=server_url, workspace_client=w)
        mcp_tools = mcp_client.list_tools()
        for mcp_tool in mcp_tools:
            input_schema = mcp_tool.inputSchema.copy() if mcp_tool.inputSchema else {}
            tool_spec = {
                "type": "function",
                "function": {
                    "name": mcp_tool.name,
                    "parameters": input_schema,
                    "description": mcp_tool.description or f"Tool: {mcp_tool.name}",
                },
            }
            exec_fn = create_managed_exec_fn(server_url, mcp_tool.name, w)
            tools.append(ToolInfo(name=mcp_tool.name, spec=tool_spec, exec_fn=exec_fn))
    return tools


# COMMAND ----------
import asyncio
from arxiv_curator.config import ProjectConfig
import nest_asyncio

nest_asyncio.apply()

cfg = ProjectConfig.from_yaml("../project_config.yml")
catalog = cfg.catalog
schema= cfg.schema
genie_space_id = cfg.genie_space_id

w = WorkspaceClient()
host = w.config.host

MANAGED_MCP_SERVER_URLS = [
    f"{host}/api/2.0/mcp/genie/{genie_space_id}",
    f"{host}/api/2.0/mcp/vector-search/{catalog}/{schema}",
]

tools = asyncio.run(
    create_mcp_tools(w=w, url_list=MANAGED_MCP_SERVER_URLS,)
)
# COMMAND ----------
