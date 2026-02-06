from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    AiGatewayConfig,
    AiGatewayInferenceTableConfig,
    EndpointCoreConfigInput,
    EndpointTag,
    ServedEntityInput,
)


def serve_model(
    entity_name: str,
    entity_version: str,
    tags: dict,
    endpoint_name: str,
    catalog_name: str,
    schema_name: str,
    table_name_prefix: str,
    scale_to_zero_enabled: bool = True,
    workload_size: str = "Small",
    environment_vars: dict | None = None,
) -> None:
    served_entities = [
        ServedEntityInput(
            entity_name=entity_name,
            scale_to_zero_enabled=scale_to_zero_enabled,
            workload_size=workload_size,
            entity_version=entity_version,
            environment_vars=environment_vars,
        )
    ]

    ai_gateway_cfg = AiGatewayConfig(
        inference_table_config=AiGatewayInferenceTableConfig(
            enabled=True,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name_prefix=table_name_prefix,
        )
    )

    workspace = WorkspaceClient()
    endpoint_exists = any(
        item.name == endpoint_name for item in workspace.serving_endpoints.list()
    )

    if not endpoint_exists:
        workspace.serving_endpoints.create(
            name=endpoint_name,
            config=EndpointCoreConfigInput(
                name=endpoint_name,
                served_entities=served_entities,
            ),
            ai_gateway=ai_gateway_cfg,
            tags=[EndpointTag.from_dict(tags)],
        )
    else:
        workspace.serving_endpoints.update_config(
            name=endpoint_name, served_entities=served_entities
        )


def call_endpoint(
    endpoint_name: str,
    messages: list[dict],
    custom_inputs: dict | None = None,
) -> dict:
    from openai import OpenAI

    workspace = WorkspaceClient()
    workspace_url = workspace.config.host
    token = workspace.tokens.create(lifetime_seconds=2000).token_value

    client = OpenAI(
        api_key=token,
        base_url=f"{workspace_url}/serving-endpoints",
    )

    response = client.responses.create(
        model=endpoint_name,
        input=messages,
        extra_body={"custom_inputs": custom_inputs} if custom_inputs else None,
    )

    return response.model_dump()
