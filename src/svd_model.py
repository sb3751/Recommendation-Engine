import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.decomposition import TruncatedSVD


# --------------------------------------------------
# Load processed train data
# --------------------------------------------------
def load_train_data():
    path = Path("data") / "processed" / "train.csv"
    return pd.read_csv(path)


# --------------------------------------------------
# Build user–item matrix
# --------------------------------------------------
def build_user_item_matrix(train):
    return train.pivot_table(
        index="user_id",
        columns="item_id",
        values="rating",
        fill_value=0
    )


# --------------------------------------------------
# Train SVD model
# --------------------------------------------------
def train_svd(user_item_matrix, n_components=50):
    svd = TruncatedSVD(
        n_components=n_components,
        random_state=42
    )

    user_latent = svd.fit_transform(user_item_matrix)
    item_latent = svd.components_

    return svd, user_latent, item_latent


# --------------------------------------------------
# Predict full rating matrix
# --------------------------------------------------
def predict_ratings(user_latent, item_latent):
    return np.dot(user_latent, item_latent)


# --------------------------------------------------
# Recommend items for a user
# --------------------------------------------------
def recommend_items_svd(
    user_id,
    user_item_matrix,
    predicted_ratings,
    top_n=5
):
    if user_id not in user_item_matrix.index:
        raise ValueError("User not found")

    user_idx = user_item_matrix.index.get_loc(user_id)

    user_ratings = user_item_matrix.iloc[user_idx].values
    user_predictions = predicted_ratings[user_idx]

    # Remove already-rated items
    user_predictions = np.where(user_ratings > 0, -np.inf, user_predictions)

    top_items_idx = np.argsort(user_predictions)[::-1][:top_n]
    item_ids = user_item_matrix.columns[top_items_idx]

    return list(zip(item_ids, user_predictions[top_items_idx]))
