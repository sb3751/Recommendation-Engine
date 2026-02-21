import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Load item metadata
# --------------------------------------------------
def load_items():
    item_path = Path("data") / "raw" / "ml-100k" / "u.item"

    items = pd.read_csv(
        item_path,
        sep="|",
        encoding="latin-1",
        header=None
    )

    return items


# --------------------------------------------------
# Assign column names
# --------------------------------------------------
def preprocess_items(items):
    items.columns = [
        "item_id",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url",
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western"
    ]

    return items


# --------------------------------------------------
# Build genre text representation
# --------------------------------------------------
def build_genre_text(items):
    genre_cols = items.columns[6:]

    items["genre_text"] = items[genre_cols].apply(
        lambda row: " ".join(
            genre for genre, val in row.items() if val == 1
        ),
        axis=1
    )

    return items


# --------------------------------------------------
# Vectorize items using TF-IDF
# --------------------------------------------------
def vectorize_items(items):
    tfidf = TfidfVectorizer()
    item_vectors = tfidf.fit_transform(items["genre_text"])
    return item_vectors


# --------------------------------------------------
# Compute cosine similarity
# --------------------------------------------------
def compute_similarity(item_vectors):
    return cosine_similarity(item_vectors)


# --------------------------------------------------
# Recommend similar items
# --------------------------------------------------
def recommend_similar_items(item_id, items, similarity_matrix, top_n=5):
    if item_id not in items["item_id"].values:
        raise ValueError(f"Item ID {item_id} not found")

    idx = items.index[items["item_id"] == item_id][0]

    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    similar_indices = [i for i, _ in similarity_scores[1: top_n + 1]]

    return items.loc[similar_indices, ["item_id", "title", "genre_text"]]


# --------------------------------------------------
# Quick test run
# --------------------------------------------------
if __name__ == "__main__":
    items = load_items()
    items = preprocess_items(items)
    items = build_genre_text(items)

    vectors = vectorize_items(items)
    similarity = compute_similarity(vectors)

    print(recommend_similar_items(1, items, similarity))
