import pandas as pd

from src.collaborative_model import (
    load_train_data,
    build_user_item_matrix,
    compute_user_similarity,
    predict_rating,
    recommend_items
)

from src.metrics import rmse, precision_at_k


def main():
    # -----------------------------
    # Load data
    # -----------------------------
    train = load_train_data()
    test = pd.read_csv("data/processed/test.csv")

    # -----------------------------
    # Build collaborative model
    # -----------------------------
    user_item = build_user_item_matrix(train)
    user_similarity = compute_user_similarity(user_item)

    # -----------------------------
    # RMSE Evaluation
    # -----------------------------
    y_true = []
    y_pred = []

    for _, row in test.iterrows():
        pred = predict_rating(
            row["user_id"],
            row["item_id"],
            user_item,
            user_similarity
        )

        if pred is not None:
            y_true.append(row["rating"])
            y_pred.append(pred)

    rmse_score = rmse(y_true, y_pred)
    print(f"RMSE: {rmse_score:.4f}")

    # -----------------------------
    # Precision@K Evaluation
    # -----------------------------
    k = 5
    precisions = []

    # LIMIT USERS to avoid long runtime (research-standard practice)
    sample_users = test["user_id"].unique()[:50]

    for user_id in sample_users:
        if user_id not in user_item.index:
            continue

        # Ground truth: relevant items (rating >= 4)
        relevant_items = test[
            (test["user_id"] == user_id) & (test["rating"] >= 4)
        ]["item_id"].tolist()

        if not relevant_items:
            continue

        recommendations = recommend_items(
            user_id,
            user_item,
            user_similarity,
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
        print(f"Precision@{k}: {avg_precision:.4f}")
    else:
        print(f"Precision@{k}: No valid users evaluated")


if __name__ == "__main__":
    main()
