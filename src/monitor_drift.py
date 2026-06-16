import os
import sys
import pandas as pd
import numpy as np

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from src.data_preprocessing import generate_synthetic_data
from src.train import load_config


def generate_drifted_production_data(num_rows: int, seed: int):
    """
    Simulate production data with intentional drift.
    """

    np.random.seed(seed + 100)

    df = generate_synthetic_data(num_rows, seed + 100)

    if "age" in df.columns:
        df["age"] = (
            df["age"].astype(float)
            + np.random.randint(5, 12, size=num_rows)
        ).clip(29, 85)

    if "cholesterol" in df.columns:
        df["cholesterol"] = (
            df["cholesterol"].astype(float)
            + np.random.randint(30, 60, size=num_rows)
        ).clip(126, 600)

    if "chest_pain_type" in df.columns:
        df["chest_pain_type"] = np.random.choice(
            [
                "typical_angina",
                "atypical_angina",
                "non_anginal",
                "asymptomatic",
            ],
            size=num_rows,
            p=[0.05, 0.15, 0.30, 0.50],
        )

    return df


def run_drift_monitoring():

    print("=" * 60)
    print("DATA DRIFT MONITORING")
    print("=" * 60)

    config = load_config()

    reference_path = "data/heart_disease.csv"

    if not os.path.exists(reference_path):
        print("Reference dataset not found.")
        print("Run: dvc pull")
        sys.exit(1)

    print("\nLoading reference dataset...")

    reference_df = pd.read_csv(reference_path)

    if "target" in reference_df.columns:
        reference_df = reference_df.drop(columns=["target"])

    print(f"Reference rows: {len(reference_df):,}")

    print("\nGenerating production dataset...")

    production_df = generate_drifted_production_data(
        config["data"]["num_rows"],
        config["data"]["random_seed"]
    )

    if "target" in production_df.columns:
        production_df = production_df.drop(columns=["target"])

    production_df = production_df[reference_df.columns]

    # Match datatypes
    for col in reference_df.columns:

        if pd.api.types.is_numeric_dtype(reference_df[col]):

            reference_df[col] = pd.to_numeric(
                reference_df[col],
                errors="coerce"
            )

            production_df[col] = pd.to_numeric(
                production_df[col],
                errors="coerce"
            )

        else:

            reference_df[col] = (
                reference_df[col]
                .fillna("missing")
                .astype(str)
            )

            production_df[col] = (
                production_df[col]
                .fillna("missing")
                .astype(str)
            )

    print("Running Evidently report...")

    report = Report(
        metrics=[
            DataDriftPreset()
        ]
    )

    report.run(
        reference_data=reference_df,
        current_data=production_df
    )

    # Save HTML report
    os.makedirs("reports", exist_ok=True)

    report_path = "reports/data_drift_report.html"

    report.save_html(report_path)

    print(f"\nHTML report saved to: {report_path}")

    # Extract summary safely
    result = report.as_dict()

    metrics = result.get("metrics", [])

    if not metrics:
        print("\nNo metrics returned by Evidently.")
        return

    drift_result = metrics[0].get("result", {})

    total_features = drift_result.get(
        "number_of_columns",
        drift_result.get("number_of_features", "Unknown")
    )

    drifted_features = drift_result.get(
        "number_of_drifted_columns",
        drift_result.get("number_of_drifted_features", "Unknown")
    )

    drift_share = drift_result.get(
        "share_of_drifted_columns",
        drift_result.get("share_of_drifted_features", 0)
    )

    print("\n" + "=" * 60)
    print("DATA DRIFT SUMMARY")
    print("=" * 60)
    print(f"Total Features Evaluated : {total_features}")
    print(f"Drifted Features         : {drifted_features}")
    print(f"Drift Percentage         : {drift_share * 100:.2f}%")
    print("=" * 60)

    # Quality gate
    DRIFT_THRESHOLD = 0.30

    if isinstance(drift_share, (int, float)):

        if drift_share > DRIFT_THRESHOLD:
            print(
                f"\nALERT: Drift exceeds threshold "
                f"({DRIFT_THRESHOLD * 100:.0f}%)"
            )
            sys.exit(1)

        print("\nSUCCESS: Drift within acceptable limits.")

    print(
        f"\nOpen report:\n"
        f"file://{os.path.abspath(report_path)}"
    )


if __name__ == "__main__":
    run_drift_monitoring()