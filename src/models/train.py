import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.data.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    prepare_features,
)

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT_DIR / "data/raw/churn.csv"
PARAMS_PATH = ROOT_DIR / "params.yaml"
MLFLOW_DB = ROOT_DIR / "mlflow.db"

MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB}"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------


def load_params():
    """Load model and training parameters."""

    with open(PARAMS_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------


def create_model_preprocessor():
    """Create preprocessing pipeline for numerical/categorical data."""

    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
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


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------


def create_models(params):
    """Create all candidate models."""

    return {
        "logistic_regression": LogisticRegression(
            C=params["models"]["logistic_regression"]["C"],
            max_iter=params["models"]["logistic_regression"]["max_iter"],
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=params["models"]["random_forest"]["n_estimators"],
            max_depth=params["models"]["random_forest"]["max_depth"],
            min_samples_split=params["models"]["random_forest"][
                "min_samples_split"
            ],
            random_state=params["models"]["random_forest"]["random_state"],
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=params["models"]["xgboost"]["n_estimators"],
            max_depth=params["models"]["xgboost"]["max_depth"],
            learning_rate=params["models"]["xgboost"]["learning_rate"],
            subsample=params["models"]["xgboost"]["subsample"],
            colsample_bytree=params["models"]["xgboost"][
                "colsample_bytree"
            ],
            random_state=params["models"]["xgboost"]["random_state"],
            eval_metric="logloss",
        ),
    }


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------


def calculate_metrics(model, X_test, y_test):
    """Calculate classification metrics."""

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    return metrics


# ---------------------------------------------------------
# MLflow parameter logging
# ---------------------------------------------------------


def log_model_parameters(
    model,
    test_size,
    random_state,
):
    """
    Log model parameters to MLflow.

    random_state is removed from model.get_params()
    because it is logged separately as an experiment-level
    parameter.
    """

    model_params = model.get_params()

    # Remove random_state to avoid duplicate logging.
    model_params.pop("random_state", None)

    # Remove parameters whose value is None.
    # This keeps the MLflow parameter table cleaner.
    model_params = {
        key: value
        for key, value in model_params.items()
        if value is not None
    }

    mlflow.log_params(model_params)

    # Experiment-level parameters.
    mlflow.log_param(
        "test_size",
        test_size,
    )

    mlflow.log_param(
        "random_state",
        random_state,
    )


# ---------------------------------------------------------
# MLflow model logging
# ---------------------------------------------------------


def log_model_to_mlflow(
    pipeline,
    model_name,
):
    """
    Log trained pipeline to MLflow.

    XGBoost contains custom Python classes that MLflow's
    skops-based model validation considers untrusted.

    Therefore, explicitly trust the XGBoost classes.
    """

    if model_name == "xgboost":

        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            skops_trusted_types=[
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ],
        )

    else:

        mlflow.sklearn.log_model(
            pipeline,
            name="model",
        )


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------


def train_models():
    """Train, evaluate and log all candidate models."""

    print("=" * 70)
    print("CHURN PREDICTION MODEL TRAINING")
    print("=" * 70)

    # -----------------------------------------------------
    # Load configuration
    # -----------------------------------------------------

    params = load_params()

    test_size = params["data"]["test_size"]
    random_state = params["data"]["random_state"]

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    # -----------------------------------------------------
    # Prepare features
    # -----------------------------------------------------

    X, y = prepare_features(df)

    print(f"Feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # -----------------------------------------------------
    # Train/test split
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print("\nTrain/Test split:")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    # -----------------------------------------------------
    # MLflow setup
    # -----------------------------------------------------

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        "churn-prediction"
    )

    # -----------------------------------------------------
    # Create models
    # -----------------------------------------------------

    models = create_models(params)

    results = []

    # -----------------------------------------------------
    # Train every candidate
    # -----------------------------------------------------

    for model_name, model in models.items():

        print("\n" + "-" * 70)
        print(f"Training: {model_name}")
        print("-" * 70)

        # -------------------------------------------------
        # Create preprocessing + model pipeline
        # -------------------------------------------------

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    create_model_preprocessor(),
                ),
                (
                    "model",
                    model,
                ),
            ]
        )

        # -------------------------------------------------
        # MLflow run
        # -------------------------------------------------

        with mlflow.start_run(
            run_name=model_name
        ) as run:

            # -------------------------------------------------
            # Train
            # -------------------------------------------------

            pipeline.fit(
                X_train,
                y_train,
            )

            # -------------------------------------------------
            # Evaluate
            # -------------------------------------------------

            metrics = calculate_metrics(
                pipeline,
                X_test,
                y_test,
            )

            # -------------------------------------------------
            # Log parameters
            # -------------------------------------------------

            log_model_parameters(
                model=model,
                test_size=test_size,
                random_state=random_state,
            )

            # -------------------------------------------------
            # Log metrics
            # -------------------------------------------------

            mlflow.log_metrics(
                metrics
            )

            # -------------------------------------------------
            # Log model
            # -------------------------------------------------

            log_model_to_mlflow(
                pipeline=pipeline,
                model_name=model_name,
            )

            # -------------------------------------------------
            # Tags
            # -------------------------------------------------

            mlflow.set_tag(
                "model_type",
                model_name,
            )

            mlflow.set_tag(
                "project",
                "mlops-churn-prediction",
            )

            # -------------------------------------------------
            # Print results
            # -------------------------------------------------

            print("\nMetrics:")

            for metric_name, value in metrics.items():

                print(
                    f"{metric_name:10s}: {value:.4f}"
                )

            # -------------------------------------------------
            # Store results
            # -------------------------------------------------

            results.append(
                {
                    "model": model_name,
                    "run_id": run.info.run_id,
                    **metrics,
                }
            )

    # ---------------------------------------------------------
    # Compare models
    # ---------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        by="f1",
        ascending=False,
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df[
            [
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
            ]
        ].to_string(index=False)
    )

    # ---------------------------------------------------------
    # Champion model
    # ---------------------------------------------------------

    champion = results_df.iloc[0]

    print("\n" + "=" * 70)
    print("CHAMPION MODEL")
    print("=" * 70)

    print(
        f"Model:   {champion['model']}"
    )

    print(
        f"F1:      {champion['f1']:.4f}"
    )

    print(
        f"ROC-AUC: {champion['roc_auc']:.4f}"
    )

    print(
        f"Run ID:  {champion['run_id']}"
    )

    # ---------------------------------------------------------
    # Save comparison results
    # ---------------------------------------------------------

    output_dir = (
        ROOT_DIR / "data/processed"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Save model comparison
    # ---------------------------------------------------------

    results_path = (
        output_dir
        / "model_comparison.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Save champion metadata
    # ---------------------------------------------------------

    champion_path = (
        output_dir
        / "champion_model.json"
    )

    with open(
        champion_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            champion.to_dict(),
            file,
            indent=4,
            default=str,
        )

    # ---------------------------------------------------------
    # Final output
    # ---------------------------------------------------------

    print(
        f"\nModel comparison saved to: "
        f"{results_path}"
    )

    print(
        f"Champion metadata saved to: "
        f"{champion_path}"
    )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------


if __name__ == "__main__":
    train_models()