from src.load_data import load_ratings
from src.preprocess import preprocess_ratings
from src.split_data import train_test_split_by_user


def main():
    ratings = load_ratings()
    print("Raw data shape:", ratings.shape)

    ratings = preprocess_ratings(ratings)
    print("After preprocessing:", ratings.shape)

    train, test = train_test_split_by_user(ratings)

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)

    train.to_csv("data/processed/train.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)

    print("Phase 1 completed successfully.")


if __name__ == "__main__":
    main()
