from src.hybrid_model import (
    build_content_similarity,
    build_svd_predictions,
    hybrid_recommend
)


def main():
    # Build components
    items, content_similarity = build_content_similarity()
    user_item, predicted_ratings = build_svd_predictions()

    # Test hybrid recommender
    sample_user = user_item.index[0]

    recommendations = hybrid_recommend(
        user_id=sample_user,
        user_item=user_item,
        predicted_ratings=predicted_ratings,
        items=items,
        content_similarity=content_similarity,
        alpha=0.7,
        top_n=5
    )

    print(f"Hybrid recommendations for user {sample_user}")
    print(recommendations)


if __name__ == "__main__":
    main()
