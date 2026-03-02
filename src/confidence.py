import numpy as np


def get_confidence_scores(model, X):
    probabilities = model.predict_proba(X)
    confidence_scores = np.max(probabilities, axis=1)
    return confidence_scores