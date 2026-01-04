import asyncio
import json
import warnings

from typing import Any
from uuid import uuid4

import backoff
import mlflow
import nest_asyncio
import openai
from databricks.sdk import WorkspaceClient
from arxiv_curator.mcp import ToolInfo, create_mcp_tools
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)


# Allow nested event loops (required for Databricks notebooks)
nest_asyncio.apply()

mlflow.set_experiment("/Shared/genai-arxiv-agent")

# COMMAND ----------
w = WorkspaceClient()

host = w.config.host
genie_space_id = "01f0c3f882e41bb9be96b4ddf295d2a4"
catalog_name = "mlops_dev"
schema_name = "arxiv"

SYSTEM_PROMPT = """You are a helpful AI assistant with access
to tools for searching arXiv papers and querying a Genie space.

When helping users:
- Use the vector search tool to find relevant arXiv papers based on semantic similarity
- Use the Genie query_space tool to answer questions about the data in the space
- Be concise and informative in your responses
- Cite paper IDs when referencing specific papers
"""





MANAGED_MCP_SERVER_URLS = [
    f"{host}/api/2.0/mcp/genie/{genie_space_id}",
    f"{host}/api/2.0/mcp/vector-search/{catalog_name}/{schema_name}",
]





tools = asyncio.run(
    create_mcp_tools(
        w=w,
        url_list=MANAGED_MCP_SERVER_URLS,
    )
)


# COMMAND ----------
class ArxivAgent(ResponsesAgent):
    def __init__(self, llm_endpoint: str, tools: list[ToolInfo]):
        """Initializes the Arxiv Agent."""
        self.llm_endpoint = llm_endpoint
        self.workspace_client = WorkspaceClient()
        self.model_serving_client = (
            self.workspace_client.serving_endpoints.get_open_ai_client()
        )
        self._tools_dict = {tool.name: tool for tool in tools}

    def get_tool_specs(self) -> list[dict]:
        """Returns tool specifications in the format OpenAI expects."""
        return [tool_info.spec for tool_info in self._tools_dict.values()]

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict) -> Any:
        """Executes the specified tool with the given arguments."""
        return self._tools_dict[tool_name].exec_fn(**args)

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    @mlflow.trace(span_type=SpanType.LLM)
    def call_llm(
        self, messages: list[dict[str, Any]]
    ) -> Generator[dict[str, Any], None, None]:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="PydanticSerializationUnexpectedValue"
            )
            for chunk in self.model_serving_client.chat.completions.create(
                model=self.llm_endpoint,
                messages=to_chat_completions_input(messages),
                tools=self.get_tool_specs(),
                stream=True,
            ):
                yield chunk.to_dict()

    def handle_tool_call(
        self, tool_call: dict[str, Any], messages: list[dict[str, Any]]
    ) -> ResponsesAgentStreamEvent:
        """
        Execute tool calls, add them to the running message history,
        and return a ResponsesStreamEvent w/ tool output
        """
        args = json.loads(tool_call["arguments"])
        result = str(self.execute_tool(tool_name=tool_call["name"], args=args))

        tool_call_output = self.create_function_call_output_item(
            tool_call["call_id"], result
        )
        messages.append(tool_call_output)
        return ResponsesAgentStreamEvent(
            type="response.output_item.done", item=tool_call_output
        )

    def call_and_run_tools(
        self,
        messages: list[dict[str, Any]],
        max_iter: int = 10,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        for _ in range(max_iter):
            last_msg = messages[-1]
            if last_msg.get("role", None) == "assistant":
                return
            elif last_msg.get("type", None) == "function_call":
                yield self.handle_tool_call(last_msg, messages)
            else:
                yield from output_to_responses_items_stream(
                    chunks=self.call_llm(messages), aggregator=messages
                )

        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(
                "Max iterations reached. Stopping.", str(uuid4())
            ),
        )

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(
            output=outputs, custom_outputs=request.custom_inputs
        )

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
            i.model_dump() for i in request.input
        ]

        yield from self.call_and_run_tools(messages)


# COMMAND ----------
# Create agent instance
llm_endpoint = "databricks-gpt-oss-120b"
agent = ArxivAgent(llm_endpoint=llm_endpoint, tools=tools)
mlflow.models.set_model(agent)
