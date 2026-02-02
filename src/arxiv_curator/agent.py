import json
import warnings
from collections.abc import Generator
from datetime import datetime
from typing import Any
from uuid import uuid4

import backoff
import mlflow
import openai
from databricks.sdk import WorkspaceClient
from mlflow import MlflowClient
from mlflow.entities import SpanType
from mlflow.models.resources import (
    DatabricksGenieSpace,
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
)
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)

from arxiv_curator.config import ProjectConfig
from arxiv_curator.mcp import ToolInfo


class ArxivAgent(ResponsesAgent):
    def __init__(self, llm_endpoint: str, tools: list[ToolInfo], system_prompt: str):
        """Initializes the Arxiv Agent."""
        self.system_prompt = system_prompt
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
        messages = [{"role": "system", "content": self.system_prompt}] + [
            i.model_dump() for i in request.input
        ]

        yield from self.call_and_run_tools(messages)

def log_register_agent(
    cfg: ProjectConfig,
    git_sha: str,
    run_id: str,
    job_id: str,
    agent_code_path: str,
    model_name: str,
    experiment_path: str = "/Shared/genai-arxiv-agent",
) -> mlflow.entities.model_registry.RegisteredModel:
    """
    Log and register an MLflow agent model to Unity Catalog.

    Args:
        cfg: Project configuration containing catalog, schema, and other settings.
        git_sha: Git commit SHA for tracking.
        run_id: Run identifier for tracking.
        job_id: Job identifier for tracking.
        model_name: Model path in Unity Catalog.
        agent_code_path: Path to the agent Python file.
        experiment_path: MLflow experiment path.

    Returns:
        RegisteredModel object from Unity Catalog.
    """


    resources = [
        DatabricksServingEndpoint(endpoint_name=cfg.llm_endpoint),
        DatabricksGenieSpace(genie_space_id=cfg.genie_space_id),
        DatabricksVectorSearchIndex(
            index_name=f"{cfg.catalog}.{cfg.schema}.arxiv_index"
        ),
    ]

    model_config = {
        "catalog": cfg.catalog,
        "schema": cfg.schema,
        "genie_space_id": cfg.genie_space_id,
        "system_prompt": cfg.system_prompt,
        "llm_endpoint": cfg.llm_endpoint,
    }

    test_request = {
    "input": [
        {"role": "user",
         "content": "What are recent papers about LLMs and reasoning?"}
    ]
    }

    mlflow.set_experiment(experiment_path)
    ts = datetime.now().strftime("%Y-%m-%d")

    with mlflow.start_run(
        run_name=f"arxiv-mcp-agent-{ts}",
        tags={"git_sha": git_sha, "run_id": run_id},
    ):
        model_info = mlflow.pyfunc.log_model(
            name="agent",
            python_model=agent_code_path,
            resources=resources,
            input_example=test_request,
            model_config=model_config,
        )

    registered_model = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=model_name,
        env_pack="databricks_model_serving"
    )

    client = MlflowClient()
    client.set_registered_model_alias(
        name=model_name,
        alias="latest-model",
        version=registered_model.version,
    )

    client.update_registered_model(
        model_name=model_name,
        deployment_job_id=job_id
    )

    return registered_model
