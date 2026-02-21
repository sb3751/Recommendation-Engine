import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def load_train_data():
    train_path = Path("data") / "processed" / "train.csv"
    train = pd.read_csv(train_path)
    return train

def build_user_item_matrix(train):
    user_item_matrix = train.pivot_table(
        index="user_id",
        columns="item_id",
        values="rating"
    )
    return user_item_matrix

def compute_user_similarity(user_item_matrix):
    filled_matrix = user_item_matrix.fillna(0)
    similarity = cosine_similarity(filled_matrix)
    similarity_df = pd.DataFrame(
        similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )
    return similarity_df

def predict_rating(user_id, item_id, user_item_matrix, user_similarity):
    if item_id not in user_item_matrix.columns:
        return None

    if user_id not in user_item_matrix.index:
        return None

    similar_users = user_similarity[user_id]
    item_ratings = user_item_matrix[item_id]

    mask = item_ratings.notna()
    if mask.sum() == 0:
        return None

    denominator = similar_users[mask].sum()
    if denominator == 0:
        return None   # <-- CRITICAL FIX

    prediction = (
        similar_users[mask] * item_ratings[mask]
    ).sum() / denominator

    return prediction


    prediction = (
        similar_users[mask] * item_ratings[mask]
    ).sum() / similar_users[mask].sum()

    return prediction

def recommend_items(user_id, user_item_matrix, user_similarity, top_n=5):
    if user_id not in user_item_matrix.index:
        raise ValueError("User not found")

    user_ratings = user_item_matrix.loc[user_id]
    unrated_items = user_ratings[user_ratings.isna()].index

    predictions = []

    for item_id in unrated_items:
        pred = predict_rating(
            user_id,
            item_id,
            user_item_matrix,
            user_similarity
        )
        if pred is not None:
            predictions.append((item_id, pred))

    predictions.sort(key=lambda x: x[1], reverse=True)

    return predictions[:top_n]

if __name__ == "__main__":
    train = load_train_data()
    print("Train data loaded:", train.shape)

    user_item = build_user_item_matrix(train)
    print("User–Item matrix shape:", user_item.shape)

    similarity = compute_user_similarity(user_item)
    print("User similarity computed")

    sample_user = user_item.index[0]
    recommendations = recommend_items(
        sample_user,
        user_item,
        similarity,
        top_n=5
    )

    print("Recommendations for user", sample_user)
    print(recommendations)
