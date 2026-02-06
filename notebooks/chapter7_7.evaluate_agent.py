# Databricks notebook source
import mlflow
from mlflow.genai.scorers import Guidelines

from arxiv_curator.agent import ArxivAgent
from arxiv_curator.config import ProjectConfig
from arxiv_curator.utils.common import set_mlflow_tracking_uri

set_mlflow_tracking_uri()

# COMMAND ----------
# Initialize the agent
cfg = ProjectConfig.from_yaml("../project_config.yml")

agent = ArxivAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt=cfg.system_prompt,
    catalog=cfg.catalog,
    schema=cfg.schema,
    genie_space_id=cfg.genie_space_id,
)

# COMMAND ----------
# Load evaluation inputs
with open("eval_inputs.txt") as f:
    eval_data = [{"inputs": {
        "question": line.strip()}} for line in f if line.strip()]


# Define predict function for the agent
def predict_fn(question: str) -> str:
    """Predict function that wraps the agent for evaluation."""
    request = {"input": [{"role": "user", "content": question}]}
    result = agent.predict(request)
    return result.output[-1].content


# COMMAND ----------

polite_tone_guideline = Guidelines(
    name="polite_tone",
    guidelines=[
        "The response must use a polite and professional tone throughout",
        "The response should be friendly and helpful without being condescending",
        "The response must avoid any dismissive or rude language"
    ],
    model="databricks:/databricks-gpt-oss-120b"
)

hook_in_post_guideline = Guidelines(
    name="hook_in_post",
    guidelines=[
        "The response must start with an engaging hook that captures attention",
        "The opening should make the reader want to continue reading",
        "The response should have a compelling introduction before diving into details"
    ],
    model="databricks:/databricks-gpt-oss-120b"
)


# COMMAND ----------
# Define code-based scorer for word count

@mlflow.genai.scorer
def word_count_check(outputs: list) -> bool:
    """Check that the output is under 350 words."""
    word_count = len(outputs[0]["text"].split())
    return word_count < 350


# COMMAND ----------
# Run evaluation
mlflow.set_experiment("/Shared/arxiv-agent-evaluation")

results = mlflow.genai.evaluate(
    predict_fn=predict_fn,
    data=eval_data,
    scorers=[word_count_check,
             polite_tone_guideline,
             hook_in_post_guideline]
)

print(results.to_pandas())

# COMMAND ----------
