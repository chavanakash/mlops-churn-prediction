import pandas as pd

from src.data.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    create_preprocessor,
    prepare_features,
)


def create_test_dataframe(include_target=True):
    """Create a small dataframe for testing."""

    data = {
        "age": [25, 40],
        "tenure": [10, 30],
        "monthly_charges": [50.0, 80.0],
        "total_charges": [500.0, 2400.0],
        "support_tickets": [1, 3],
        "senior_citizen": [0, 1],
        "contract": ["One year", "Month-to-month"],
        "internet_service": ["DSL", "Fiber optic"],
        "payment_method": [
            "Credit card",
            "Electronic check",
        ],
        "tech_support": ["Yes", "No"],
        "partner": ["Yes", "No"],
        "dependents": ["No", "Yes"],
    }

    if include_target:
        data["churn"] = [0, 1]

    return pd.DataFrame(data)


def test_prepare_features():
    """Test feature/target separation."""

    df = create_test_dataframe()

    X, y = prepare_features(df)

    expected_features = (
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    )

    assert list(X.columns) == expected_features
    assert len(X) == 2
    assert len(y) == 2


def test_preprocessor_transforms_data():
    """Test that the preprocessing pipeline transforms data."""

    df = create_test_dataframe(include_target=False)

    preprocessor = create_preprocessor()

    transformed = preprocessor.fit_transform(df)

    # Same number of rows after transformation
    assert transformed.shape[0] == len(df)

    # We should have more features than the original
    # numerical features because categorical features
    # are one-hot encoded.
    assert transformed.shape[1] > len(NUMERIC_FEATURES)


def test_unknown_category_does_not_crash():
    """
    Test that unseen categorical values do not crash
    the preprocessing pipeline.
    """

    train_df = create_test_dataframe(
        include_target=False
    )

    new_data = train_df.copy()

    # Introduce a category that was not present
    # during preprocessing.fit()
    new_data.loc[0, "contract"] = "Annual Premium"

    preprocessor = create_preprocessor()

    # Fit only on training data
    preprocessor.fit(train_df)

    # Transform data containing an unseen category
    transformed = preprocessor.transform(new_data)

    assert transformed.shape[0] == len(new_data)