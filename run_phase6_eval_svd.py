import pandas as pd
import numpy as np

from src.svd_model import (
    load_train_data,
    build_user_item_matrix,
    train_svd,
    predict_ratings,
    recommend_items_svd
)

from src.metrics import rmse, precision_at_k


def main():
    # -----------------------------
    # Load data
    # -----------------------------
    train = load_train_data()
    test = pd.read_csv("data/processed/test.csv")

    # -----------------------------
    # Train SVD model
    # -----------------------------
    user_item = build_user_item_matrix(train)
    svd, user_latent, item_latent = train_svd(user_item, n_components=50)
    predicted_ratings = predict_ratings(user_latent, item_latent)

    # -----------------------------
    # RMSE Evaluation
    # -----------------------------
    y_true = []
    y_pred = []

    for _, row in test.iterrows():
        user_id = row["user_id"]
        item_id = row["item_id"]

        if user_id not in user_item.index:
            continue
        if item_id not in user_item.columns:
            continue

        user_idx = user_item.index.get_loc(user_id)
        item_idx = user_item.columns.get_loc(item_id)

        y_true.append(row["rating"])
        y_pred.append(predicted_ratings[user_idx, item_idx])

    rmse_score = rmse(y_true, y_pred)
    print(f"SVD RMSE: {rmse_score:.4f}")

    # -----------------------------
    # Precision@K Evaluation
    # -----------------------------
    k = 5
    precisions = []

    # sample users for speed (research-standard)
    sample_users = test["user_id"].unique()[:50]

    for user_id in sample_users:
        if user_id not in user_item.index:
            continue

        relevant_items = test[
            (test["user_id"] == user_id) & (test["rating"] >= 4)
        ]["item_id"].tolist()

        if not relevant_items:
            continue

        recommendations = recommend_items_svd(
            user_id,
            user_item,
            predicted_ratings,
            top_n=k
        )

        recommended_items = [item for item, _ in recommendations]

        precision = precision_at_k(
            recommended_items,
            relevant_items,
            k
        )

        precisions.append(precision)

    if precisions:
        avg_precision = sum(precisions) / len(precisions)
        print(f"SVD Precision@{k}: {avg_precision:.4f}")
    else:
        print(f"SVD Precision@{k}: No valid users evaluated")


if __name__ == "__main__":
    main()
