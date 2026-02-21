from src.svd_model import (
    load_train_data,
    build_user_item_matrix,
    train_svd,
    predict_ratings,
    recommend_items_svd
)


def main():
    train = load_train_data()
    print("Train loaded:", train.shape)

    user_item = build_user_item_matrix(train)
    print("User–Item matrix:", user_item.shape)

    svd, user_latent, item_latent = train_svd(user_item, n_components=50)
    print("SVD trained")

    predicted_ratings = predict_ratings(user_latent, item_latent)
    print("Rating matrix reconstructed")

    sample_user = user_item.index[0]
    recommendations = recommend_items_svd(
        sample_user,
        user_item,
        predicted_ratings,
        top_n=5
    )

    print(f"Top recommendations for user {sample_user}")
    print(recommendations)


if __name__ == "__main__":
    main()
