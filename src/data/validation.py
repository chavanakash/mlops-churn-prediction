"""
Data validation module.

Validates schema, data types, missing values, ranges,
duplicates, and target values before preprocessing.
"""


import pandas as pd

from src.data.ingestion import ingest_data, load_config

EXPECTED_COLUMNS = [
    "customer_id",
    "age",
    "tenure",
    "monthly_charges",
    "total_charges",
    "contract",
    "internet_service",
    "payment_method",
    "support_tickets",
    "tech_support",
    "senior_citizen",
    "partner",
    "dependents",
    "churn",
]


NUMERIC_COLUMNS = [
    "age",
    "tenure",
    "monthly_charges",
    "total_charges",
    "support_tickets",
    "senior_citizen",
    "churn",
]


CATEGORICAL_COLUMNS = [
    "contract",
    "internet_service",
    "payment_method",
    "tech_support",
    "partner",
    "dependents",
]


def validate_columns(df: pd.DataFrame) -> None:
    """Validate that the dataset contains the expected columns."""

    actual_columns = list(df.columns)

    missing_columns = set(EXPECTED_COLUMNS) - set(actual_columns)
    unexpected_columns = set(actual_columns) - set(EXPECTED_COLUMNS)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    if unexpected_columns:
        raise ValueError(
            f"Unexpected columns: {sorted(unexpected_columns)}"
        )


def validate_missing_values(df: pd.DataFrame) -> None:
    """Validate missing values."""

    missing_counts = df.isnull().sum()

    columns_with_missing_values = (
        missing_counts[missing_counts > 0]
    )

    if not columns_with_missing_values.empty:
        raise ValueError(
            "Missing values detected:\n"
            f"{columns_with_missing_values}"
        )


def validate_duplicates(df: pd.DataFrame) -> None:
    """Validate duplicate customer records."""

    duplicate_rows = df.duplicated().sum()

    if duplicate_rows > 0:
        raise ValueError(
            f"Found {duplicate_rows} duplicate rows."
        )

    duplicate_customer_ids = (
        df["customer_id"].duplicated().sum()
    )

    if duplicate_customer_ids > 0:
        raise ValueError(
            "Duplicate customer_id values detected."
        )


def validate_numeric_ranges(df: pd.DataFrame) -> None:
    """Validate numerical feature ranges."""

    if not df["age"].between(18, 100).all():
        raise ValueError(
            "Invalid age values detected."
        )

    if not df["tenure"].between(1, 72).all():
        raise ValueError(
            "Invalid tenure values detected."
        )

    if not (df["monthly_charges"] > 0).all():
        raise ValueError(
            "monthly_charges must be greater than zero."
        )

    if not (df["total_charges"] >= 0).all():
        raise ValueError(
            "total_charges cannot be negative."
        )

    if not (df["support_tickets"] >= 0).all():
        raise ValueError(
            "support_tickets cannot be negative."
        )


def validate_categorical_values(df: pd.DataFrame) -> None:
    """Validate categorical feature values."""

    allowed_values = {
        "contract": {
            "Month-to-month",
            "One year",
            "Two year",
        },
        "internet_service": {
            "DSL",
            "Fiber optic",
            "No",
        },
        "payment_method": {
            "Electronic check",
            "Mailed check",
            "Bank transfer",
            "Credit card",
        },
        "tech_support": {
            "Yes",
            "No",
        },
        "partner": {
            "Yes",
            "No",
        },
        "dependents": {
            "Yes",
            "No",
        },
    }

    for column, allowed in allowed_values.items():
        invalid_values = set(df[column].unique()) - allowed

        if invalid_values:
            raise ValueError(
                f"Invalid values found in {column}: "
                f"{sorted(invalid_values)}"
            )


def validate_target(df: pd.DataFrame) -> None:
    """Validate target variable."""

    unique_targets = set(df["churn"].unique())

    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            f"Invalid churn values: {unique_targets}"
        )


def validate_data(df: pd.DataFrame) -> bool:
    """Run all data validation checks."""

    print("Running data validation...")

    validate_columns(df)

    validate_missing_values(df)

    validate_duplicates(df)

    validate_numeric_ranges(df)

    validate_categorical_values(df)

    validate_target(df)

    print("Data validation passed successfully.")

    return True


def main() -> None:
    """Run data ingestion followed by validation."""

    config = load_config()

    df = ingest_data(config)

    validate_data(df)


if __name__ == "__main__":
    main()