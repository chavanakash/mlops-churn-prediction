"""
Data preprocessing module.

Creates a reproducible scikit-learn preprocessing pipeline
for numerical and categorical features.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.ingestion import ingest_data, load_config

CONFIG_PATH = Path("configs/data.yaml")

PREPROCESSOR_PATH = Path(
    "data/processed/preprocessor.joblib"
)


NUMERIC_FEATURES = [
    "age",
    "tenure",
    "monthly_charges",
    "total_charges",
    "support_tickets",
    "senior_citizen",
]


CATEGORICAL_FEATURES = [
    "contract",
    "internet_service",
    "payment_method",
    "tech_support",
    "partner",
    "dependents",
]


def create_preprocessor() -> ColumnTransformer:
    """
    Create the preprocessing pipeline.

    Numerical features:
        - median imputation
        - standard scaling

    Categorical features:
        - most-frequent imputation
        - one-hot encoding
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return preprocessor


def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from target."""

    X = df[
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    ].copy()

    y = df["churn"].copy()

    return X, y


def fit_preprocessor(
    X: pd.DataFrame,
) -> ColumnTransformer:
    """Fit preprocessing pipeline on training data."""

    preprocessor = create_preprocessor()

    preprocessor.fit(X)

    return preprocessor


def save_preprocessor(
    preprocessor: ColumnTransformer,
    output_path: Path = PREPROCESSOR_PATH,
) -> None:
    """Save fitted preprocessing pipeline."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        preprocessor,
        output_path,
    )

    print(
        f"Preprocessor saved to: {output_path}"
    )


def main() -> None:
    """Run preprocessing."""

    config = load_config()

    df = ingest_data(config)

    X, y = prepare_features(df)

    preprocessor = fit_preprocessor(X)

    X_transformed = preprocessor.transform(X)

    print(
        f"Original feature shape: {X.shape}"
    )

    print(
        f"Transformed feature shape: "
        f"{X_transformed.shape}"
    )

    print(
        f"Target shape: {y.shape}"
    )

    save_preprocessor(preprocessor)


if __name__ == "__main__":
    main()