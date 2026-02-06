from mlflow import MlflowClient

from arxiv_curator.agent import log_register_agent
from arxiv_curator.config import ProjectConfig
from arxiv_curator.utils.common import create_parser

args = create_parser()

root_path = args.root_path
config_path = f"{root_path}/files/project_config.yml"
agent_code_path = f"{root_path}/files/project_config.yml"

cfg = ProjectConfig.from_yaml(config_path=config_path,
                                         env=args.env)

model_name = f"{cfg.catalog}.{cfg.schema}.arxiv_agent"

registered_model = log_register_agent(
    git_sha=args.git_sha,
    run_id=args.run_id,
    agent_code_path=agent_code_path,
    model_name = model_name)


client = MlflowClient()
client.set_registered_model_alias(
    name=model_name,
    alias="latest-model",
    version=registered_model.version,
)

client.update_registered_model(
    model_name=model_name,
    deployment_job_id=args.job_id
)
