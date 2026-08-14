"""
Data ingestion module.

Responsible for loading raw customer churn data from the configured source.
"""

from pathlib import Path

import pandas as pd
import yaml

CONFIG_PATH = Path("configs/data.yaml")


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load data pipeline configuration."""

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ingest_data(config: dict) -> pd.DataFrame:
    """Load the raw churn dataset."""

    data_config = config["data"]

    source = data_config["source"]
    raw_path = Path(data_config["raw_path"])

    if source != "local":
        raise ValueError(
            f"Unsupported data source: {source}"
        )

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {raw_path}"
        )

    df = pd.read_csv(raw_path)

    if df.empty:
        raise ValueError("Raw dataset is empty.")

    print(f"Loaded dataset from: {raw_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    return df


def main() -> None:
    """Run the ingestion pipeline."""

    config = load_config()

    df = ingest_data(config)

    print("\nDataset preview:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)


if __name__ == "__main__":
    main()