import pandas as pd
from pathlib import Path


# File paths
RAW_DATA_PATH = Path(
    "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

PROCESSED_DATA_PATH = Path(
    "data/processed/cleaned_telco_churn.csv"
)


def preprocess_data():
    print("Loading raw dataset...")

    # Load the dataset
    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Original dataset shape: {df.shape}")

    # Remove extra spaces from column names
    df.columns = df.columns.str.strip()

    # Convert TotalCharges from text to numeric
    # Invalid or blank values become NaN
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Fill missing TotalCharges values with the median
    missing_total_charges = df["TotalCharges"].isnull().sum()

    if missing_total_charges > 0:
        print(
            f"Missing TotalCharges values: "
            f"{missing_total_charges}"
        )

        df["TotalCharges"] = df["TotalCharges"].fillna(
            df["TotalCharges"].median()
        )

    # Convert target column: No = 0, Yes = 1
    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    # Remove customerID because it is only an identifier
    df = df.drop(columns=["customerID"])

    # Create the processed-data folder if it does not exist
    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save the cleaned dataset
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Cleaned dataset shape: {df.shape}")
    print(f"Cleaned dataset saved to: {PROCESSED_DATA_PATH}")
    print("\nMissing values after preprocessing:")
    print(df.isnull().sum())


if __name__ == "__main__":
    preprocess_data()