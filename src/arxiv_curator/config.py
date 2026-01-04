from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ProjectConfig:
    """Configuration for the arxiv curator project."""

    catalog_name: str
    schema_name: str
    volume_name: str
    genie_space_id: str
    llm_endpoint: str

    @classmethod
    def from_yaml(cls, config_path: str | Path, env: str = "dev") -> "ProjectConfig":
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to the project_config.yml file
            env: Environment name (dev, acc, prd)

        Returns:
            ProjectConfig instance
        """
        if env not in ["prd", "acc", "dev"]:
            raise ValueError(
                f"Invalid environment: {env}. Expected 'prd', 'acc', or 'dev'"
            )

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        config_dict = config_dict[env]
        return cls(**config_dict)

    def get_full_table_name(self, table_attr: str) -> str:
        """
        Get fully qualified table name.

        Args:
            table_attr: Attribute name for the table (e.g., 'arxiv_papers_table')

        Returns:
            Fully qualified table name (catalog.schema.table)
        """
        table_name = getattr(self, table_attr)
        return f"{self.catalog_name}.{self.schema_name}.{table_name}"
