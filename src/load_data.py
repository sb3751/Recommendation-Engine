import pandas as pd
from pathlib import Path


def load_ratings():
    data_path = Path("data") / "raw" / "ml-100k" / "u.data"

    ratings = pd.read_csv(
        data_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"]
    )

    return ratings
