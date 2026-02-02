# Databricks notebook source
import mlflow
from mlflow.genai.scorers import Guidelines

# Example using Guidelines for escalation handling
mlflow.set_experiment("/Shared/guidelines-example")
escalation_guidelines = Guidelines(
    name="escalation_handling",
    guidelines=[
        "When a user indicates previous attempts failed, the response must "
        "acknowledge their efforts and either escalate or offer a new approach"
    ],
    model="databricks"
)

data = [
    {
        "inputs": {"message": "Tried everything you suggested"},
        "outputs": "Have you tried restarting?",
    },
    {
        "inputs": {"message": "Tried everything you suggested"},
        "outputs": (
            "I understand you've already tried the previous suggestions "
            "without success. Let me escalate this to our senior support "
            "team who can look into this more deeply. In the meantime, "
            "could you share any error logs you've encountered?"
        ),
    },
]

mlflow.genai.evaluate(data=data, scorers=[escalation_guidelines])

# COMMAND ----------
import mlflow
from mlflow.genai.judges import make_judge

# Example using make_judge for escalation quality evaluation (1-5 scale)
mlflow.set_experiment("/Shared/make-judge-example")
escalation_judge = make_judge(
    name="escalation_quality",
    instructions=(
        "Evaluate how well the response in {{ outputs }} handles the user's "
        "message in {{ inputs }} when they indicate previous attempts failed. "
        "Score from 1 to 5:\n"
        "1 - Ignores context, repeats already-tried suggestions\n"
        "2 - Acknowledges frustration but offers no new solutions\n"
        "3 - Offers a new approach but lacks empathy or clarity\n"
        "4 - Acknowledges efforts and offers a reasonable new approach\n"
        "5 - Empathetic, escalates appropriately or provides creative solution"
    ),
    model="databricks",
    feedback_value_type=int,
)

mlflow.genai.evaluate(data=data, scorers=[escalation_judge])

# COMMAND ----------

import mlflow
from mlflow.genai.judges import make_judge
from typing import Literal
import time

performance_judge = make_judge(
    name="performance_analyzer",
    instructions=(
        "Analyze the {{ trace }} for performance issues.\n\n"
        "Check for:\n"
        "- Operations taking longer than 2 seconds\n"
        "- Redundant API calls or database queries\n"
        "- Inefficient data processing patterns\n"
        "- Proper use of caching mechanisms\n\n"
        "Rate as: 'optimal', 'acceptable', or 'needs_improvement'"
    ),
    feedback_value_type=Literal["optimal", "acceptable", "needs_improvement"],
    model="databricks",
)

# COMMAND ----------

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

