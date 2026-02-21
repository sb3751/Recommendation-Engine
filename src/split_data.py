import pandas as pd


def train_test_split_by_user(ratings, train_ratio=0.8):
    train_parts = []
    test_parts = []

    for _, user_group in ratings.groupby("user_id"):
        split_point = int(len(user_group) * train_ratio)

        train_parts.append(user_group.iloc[:split_point])
        test_parts.append(user_group.iloc[split_point:])

    train = pd.concat(train_parts)
    test = pd.concat(test_parts)

    return train, test
