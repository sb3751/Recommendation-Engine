import numpy as np


def rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def precision_at_k(recommended_items, relevant_items, k):
    if k == 0:
        return 0.0

    recommended_k = recommended_items[:k]
    relevant_set = set(relevant_items)

    hits = sum(1 for item in recommended_k if item in relevant_set)
    return hits / k
