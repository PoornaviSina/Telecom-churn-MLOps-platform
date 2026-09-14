import pandas as pd
import json

from pathlib import Path
from scipy.stats import ks_2samp


REFERENCE_DATA = Path(
    "data/processed/cleaned_telco_churn.csv"
)

CURRENT_DATA = Path(
    "data/current/current_telco_churn.csv"
)

REPORT_PATH = Path(
    "monitoring/drift_report.json"
)


def detect_drift():

    print("Loading reference dataset...")
    reference_df = pd.read_csv(REFERENCE_DATA)

    print("Loading current dataset...")
    current_df = pd.read_csv(CURRENT_DATA)

    numeric_features = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    results = []
    drift_detected = False

    print("\nDrift Detection Results")
    print("-----------------------")

    for feature in numeric_features:

        statistic, p_value = ks_2samp(
            reference_df[feature],
            current_df[feature]
        )

        if p_value < 0.05:
            status = "DRIFT DETECTED"
            drift_detected = True
        else:
            status = "No significant drift"

        print(f"\nFeature: {feature}")
        print(f"KS Statistic: {statistic:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Status: {status}")

        results.append({
            "feature": feature,
            "ks_statistic": round(float(statistic), 4),
            "p_value": round(float(p_value), 4),
            "status": status
        })

    overall_status = (
        "DRIFT DETECTED"
        if drift_detected
        else "NO SIGNIFICANT DRIFT"
    )

    print("\nOverall Drift Status")
    print("--------------------")

    if drift_detected:
        print("WARNING: Data drift detected.")
    else:
        print("No significant data drift detected.")

    report = {
        "overall_status": overall_status,
        "features": results
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(REPORT_PATH, "w") as file:
        json.dump(
            report,
            file,
            indent=4
        )

    print(f"\nDrift report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    detect_drift()