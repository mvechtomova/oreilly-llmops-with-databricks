# Databricks notebook source
import random
from datetime import datetime

from arxiv_curator.serving import call_endpoint

endpoint_name = "arxiv-agent-endpoint"



timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
session_id = f"s-{timestamp}-{random.randint(100000, 999999)}"
request_id = f"req-{timestamp}-{random.randint(100000, 999999)}"

response = call_endpoint(
    endpoint_name=endpoint_name,
    messages=[
        {"role": "user", "content": "What are recent papers about LLMs and reasoning?"}
    ],
    custom_inputs={
        "session_id": session_id,
        "request_id": request_id,
    },
)
print(response)
