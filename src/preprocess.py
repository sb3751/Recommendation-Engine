def preprocess_ratings(ratings):
    # sort by time to prevent future data leakage
    ratings = ratings.sort_values("timestamp")

    # safety check: valid ratings only
    ratings = ratings[ratings["rating"].between(1, 5)]

    return ratings
