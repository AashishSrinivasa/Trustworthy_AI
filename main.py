import pandas as pd

from src.model import train_model
from src.confidence import get_confidence_scores
from src.drift import detect_drift
from src.trust_logic import trust_decision


def main():
    # Load data
    df = pd.read_csv("data/loan_data.csv")

    # Train model
    model, X_train, X_test, y_train, y_test = train_model(df)

    # Confidence scores
    confidence_scores = get_confidence_scores(model, X_test)

    # Drift detection
    drift_df = detect_drift(X_train, X_test)
    drift_exists = drift_df["drift_detected"].any()

    print("\n--- DATA DRIFT REPORT ---")
    print(drift_df)
    print("\nDrift detected:", drift_exists)

    # Trust decisions
    decisions = [
        trust_decision(conf, drift_exists)
        for conf in confidence_scores
    ]

    result_df = pd.DataFrame({
        "confidence": confidence_scores,
        "final_decision": decisions
    })

    print("\n--- FINAL TRUST DECISIONS (Sample) ---")
    print(result_df.head(10))


if __name__ == "__main__":
    main()