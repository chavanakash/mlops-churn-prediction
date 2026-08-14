"""
Generate a synthetic but realistic customer churn dataset.

Usage:
    python scripts/generate_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
N_CUSTOMERS = 5000

RAW_DATA_DIR = Path("data/raw")
RAW_DATA_PATH = RAW_DATA_DIR / "churn.csv"


def generate_customer_data(
    n_customers: int = N_CUSTOMERS,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate synthetic customer churn data."""

    rng = np.random.default_rng(random_seed)

    customer_ids = [
        f"CUST_{i:06d}"
        for i in range(1, n_customers + 1)
    ]

    age = rng.integers(18, 75, size=n_customers)

    tenure = rng.integers(1, 73, size=n_customers)

    monthly_charges = np.round(
        rng.uniform(20, 120, size=n_customers),
        2,
    )

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_customers,
        p=[0.55, 0.25, 0.20],
    )

    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"],
        size=n_customers,
        p=[0.35, 0.45, 0.20],
    )

    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer",
            "Credit card",
        ],
        size=n_customers,
        p=[0.35, 0.20, 0.25, 0.20],
    )

    support_tickets = rng.poisson(
        lam=2.5,
        size=n_customers,
    )

    tech_support = rng.choice(
        ["Yes", "No"],
        size=n_customers,
        p=[0.35, 0.65],
    )

    senior_citizen = rng.choice(
        [0, 1],
        size=n_customers,
        p=[0.84, 0.16],
    )

    partner = rng.choice(
        ["Yes", "No"],
        size=n_customers,
        p=[0.48, 0.52],
    )

    dependents = rng.choice(
        ["Yes", "No"],
        size=n_customers,
        p=[0.30, 0.70],
    )

    total_charges = np.round(
        monthly_charges * tenure * rng.uniform(
            0.90,
            1.10,
            size=n_customers,
        ),
        2,
    )

    # ------------------------------------------------------------------
    # Generate churn probability.
    #
    # Churn is intentionally influenced by realistic business factors:
    # - Short tenure → higher churn
    # - High monthly charges → higher churn
    # - Month-to-month contract → higher churn
    # - More support tickets → higher churn
    # - Fiber optic → slightly higher churn
    # - Lack of tech support → higher churn
    # - Senior citizens → slightly higher churn
    # ------------------------------------------------------------------

    churn_score = (
        -2.8
        + 0.035 * (monthly_charges - 60)
        - 0.035 * (tenure - 24)
        + 1.15 * (contract == "Month-to-month")
        + 0.22 * support_tickets
        + 0.45 * (internet_service == "Fiber optic")
        + 0.40 * (tech_support == "No")
        + 0.30 * senior_citizen
        - 0.25 * (partner == "Yes")
        - 0.20 * (dependents == "Yes")
    )

    churn_probability = 1 / (
        1 + np.exp(-churn_score)
    )

    churn = rng.binomial(
        n=1,
        p=churn_probability,
    )

    data = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "age": age,
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract": contract,
            "internet_service": internet_service,
            "payment_method": payment_method,
            "support_tickets": support_tickets,
            "tech_support": tech_support,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "churn": churn,
        }
    )

    return data


def save_dataset(data: pd.DataFrame, output_path: Path) -> None:
    """Save dataset to disk."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        output_path,
        index=False,
    )

    print(f"Dataset saved to: {output_path}")
    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")
    print(f"Churn rate: {data['churn'].mean():.2%}")


def main() -> None:
    """Generate and save the dataset."""

    data = generate_customer_data()

    save_dataset(
        data=data,
        output_path=RAW_DATA_PATH,
    )


if __name__ == "__main__":
    main()