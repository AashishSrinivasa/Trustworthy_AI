def trust_decision(confidence, drift_detected):
    if confidence >= 0.80 and not drift_detected:
        return "AUTO_APPROVE"
    elif confidence >= 0.60:
        return "REVIEW_REQUIRED"
    else:
        return "DO_NOT_TRUST"