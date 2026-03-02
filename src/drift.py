import pandas as pd


def detect_drift(reference_data, current_data, threshold=0.5):
    drift_results = []

    for col in reference_data.columns:
        ref_mean = reference_data[col].mean()
        cur_mean = current_data[col].mean()
        ref_std = reference_data[col].std()

        mean_diff = abs(ref_mean - cur_mean)
        drift_detected = mean_diff > (threshold * ref_std)

        drift_results.append({
            "feature": col,
            "mean_diff": mean_diff,
            "drift_detected": drift_detected
        })

    return pd.DataFrame(drift_results)