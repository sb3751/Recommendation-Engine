import numpy as np
import pandas as pd

from src.content_model import (
    load_items,
    preprocess_items,
    build_genre_text,
    vectorize_items,
    compute_similarity
)

from src.svd_model import (
    load_train_data,
    build_user_item_matrix,
    train_svd,
    predict_ratings
)


# --------------------------------------------------
# Build content-based similarity (aligned by item_id)
# --------------------------------------------------
def build_content_similarity():
    items = load_items()
    items = preprocess_items(items)
    items = build_genre_text(items)

    item_vectors = vectorize_items(items)
    similarity = compute_similarity(item_vectors)

    # index by item_id for alignment
    similarity_df = pd.DataFrame(
        similarity,
        index=items["item_id"],
        columns=items["item_id"]
    )

    return items, similarity_df


# --------------------------------------------------
# Build SVD predicted ratings
# --------------------------------------------------
def build_svd_predictions():
    train = load_train_data()
    user_item = build_user_item_matrix(train)

    _, user_latent, item_latent = train_svd(user_item, n_components=50)
    predicted_ratings = predict_ratings(user_latent, item_latent)

    predicted_df = pd.DataFrame(
        predicted_ratings,
        index=user_item.index,
        columns=user_item.columns
    )

    return user_item, predicted_df


# --------------------------------------------------
# Hybrid recommendation (ALIGNED & SAFE)
# --------------------------------------------------
def hybrid_recommend(
    user_id,
    user_item,
    predicted_ratings,
    items,
    content_similarity,
    alpha=0.7,
    top_n=5
):
    if user_id not in user_item.index:
        raise ValueError("User not found")

    # ---------- Collaborative scores ----------
    svd_scores = predicted_ratings.loc[user_id].copy()
    svd_scores[user_item.loc[user_id] > 0] = -np.inf

    # ---------- Content scores (aligned by item_id) ----------
    liked_items = user_item.loc[user_id]
    liked_items = liked_items[liked_items >= 4].index.tolist()

    if liked_items:
        content_scores = content_similarity.loc[liked_items].mean()
        # align to SVD items
        content_scores = content_scores.reindex(svd_scores.index).fillna(0)
    else:
        content_scores = pd.Series(
            0, index=svd_scores.index
        )

    # ---------- Hybrid fusion ----------
    hybrid_scores = alpha * svd_scores + (1 - alpha) * content_scores

    top_items = hybrid_scores.sort_values(ascending=False).head(top_n)

    return list(zip(top_items.index.tolist(), top_items.values.tolist()))
