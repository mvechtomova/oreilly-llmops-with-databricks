# Databricks notebook source
# https://docs.databricks.com/aws/en/generative-ai/mcp/managed-mcp
# https://docs.databricks.com/aws/en/mlflow3/genai/tracing/prod-tracing

from databricks.sdk import WorkspaceClient
import arxiv
import json
import os
from typing import List
from databricks_mcp import DatabricksMCPClient
from pydantic import BaseModel
from collections.abc import Callable
import asyncio
import nest_asyncio
import mlflow
from dotenv import load_dotenv

# Allow nested event loops (required for Databricks notebooks)
nest_asyncio.apply()
# COMMAND ----------
if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
        load_dotenv()
        profile = os.environ["PROFILE"]
        mlflow.set_tracking_uri(f"databricks://{profile}")
        mlflow.set_registry_uri(f"databricks-uc://{profile}")

PAPER_DIR = "papers"
mlflow.set_experiment("/Shared/genai-arxiv-agent")
# COMMAND ----------
w = WorkspaceClient()
openai_client = w.serving_endpoints.get_open_ai_client()

host = w.config.host
genie_space_id = "01f0c3f882e41bb9be96b4ddf295d2a4"
catalog_name = "mlops_dev"
schema_name = "arxiv"

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

MANAGED_MCP_SERVER_URLS = [f"{host}/api/2.0/mcp/genie/{genie_space_id}",
                           f"{host}/api/2.0/mcp/vector-search/{catalog_name}/{schema_name}"]


def create_managed_exec_fn(server_url: str, tool_name: str, w: WorkspaceClient) -> Callable:
    def exec_fn(**kwargs):
        client = DatabricksMCPClient(server_url=server_url, workspace_client=w)
        response = client.call_tool(tool_name, kwargs)
        return "".join([c.text for c in response.content])
    return exec_fn


async def create_mcp_tools(w: WorkspaceClient, url_list: List[str]) -> List[ToolInfo]:
    tools = []
    for server_url in url_list:
        mcp_client = DatabricksMCPClient(server_url=server_url, workspace_client=w)
        mcp_tools = mcp_client.list_tools()
        for mcp_tool in mcp_tools:
            # Get the input schema and remove conversation_id if present
            input_schema = mcp_tool.inputSchema.copy() if mcp_tool.inputSchema else {}

            # # Remove conversation_id from properties and required fields
            # if "properties" in input_schema and "conversation_id" in input_schema["properties"]:
            #     input_schema["properties"] = {k: v for k, v in input_schema["properties"].items() if k != "conversation_id"}

            # if "required" in input_schema and "conversation_id" in input_schema["required"]:
            #     input_schema["required"] = [r for r in input_schema["required"] if r != "conversation_id"]

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

tools = asyncio.run(
    create_mcp_tools(
        w=w,
        url_list=MANAGED_MCP_SERVER_URLS,
    )
)

# COMMAND ----------
@mlflow.trace(span_type="TOOL", name="execute_tool")
def execute_tool(tool_name: str, tool_args: dict) -> str:
    """
    Execute a tool by name with the given arguments.

    Args:
        tool_name: Name of the tool to execute
        tool_args: Dictionary of arguments to pass to the tool

    Returns:
        String result from the tool execution
    """
    for tool in tools:
        if tool.name == tool_name:
            result = tool.exec_fn(**tool_args)
            return result

    raise ValueError(f"Tool '{tool_name}' not found in available tools")

# COMMAND ----------
@mlflow.trace(span_type="LLM", name="call_llm")
def call_llm(messages: list, tools_specs: list):
    """
    Call the LLM with messages and tool specifications.

    Args:
        messages: List of conversation messages
        tools_specs: List of tool specifications for the LLM

    Returns:
        LLM response
    """
    response = openai_client.chat.completions.create(
        max_tokens=2024,
        model='databricks-gpt-oss-120b',
        tools=tools_specs,
        messages=messages
    )

    return response

# COMMAND ----------
@mlflow.trace(span_type="AGENT", name="process_query")
def process_query(query, max_iterations=8):
    messages = [{'role': 'user', 'content': query}]
    tools_specs = [tool.spec for tool in tools]

    response = call_llm(messages, tools_specs)

    iteration_count = 0
    process_query = True

    while process_query:
        # Safety check: prevent infinite loops
        iteration_count += 1
        if iteration_count > max_iterations:
            print(f"""\n Maximum iteration limit ({max_iterations}) reached. 
                  Sorry, I cannot help with this request 
                  as it requires too many tool calls.""")
            break

        # For OpenAI client, access the message from choices
        message = response.choices[0].message

        # Check if there are tool calls
        if message.tool_calls:
            # Add assistant message to conversation
            messages.append({
                'role': 'assistant',
                'content': message.content,
                'tool_calls': message.tool_calls
            })

            # Process each tool call
            for tool_call in message.tool_calls:
                tool_id = tool_call.id
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                print(f"[Iteration {iteration_count}] Calling tool {tool_name} with args {tool_args}")

                result = execute_tool(tool_name, tool_args)

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result
                })

            # Get next response with tool results
            response = call_llm(messages, tools_specs)
        else:
            # No tool calls, just print the response and exit
            if message.content:
                print(message.content)
            process_query = False

# COMMAND ----------
def chat_loop():
    print("Type your queries or 'quit' to exit.")
    while True:
        try:
            query = input("\nQuery: ").strip()
            if query.lower() == 'quit':
                break
    
            process_query(query)
            print("\n")
        except Exception as e:
            print(f"\nError: {str(e)}")
# COMMAND ----------
chat_loop()
# COMMAND ----------
