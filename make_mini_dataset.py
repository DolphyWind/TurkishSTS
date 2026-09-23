from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


DATA_PATH: Path = Path("/run/media/yunus/HDD_5TB/data/NLP/vngrs-web-corpus-sentences/train-00000-of-00284_sat.parquet")
OUT_BASE_PATH: Path = Path("data/")

TRAIN_SIZE: int = 100_000
VAL_SIZE: int = 10_000
SEED: int = 42


def main():
    df: pd.DataFrame = pd.read_parquet(DATA_PATH)
    groups = df["source"]

    gss = GroupShuffleSplit(
        n_splits=1,
        train_size=0.8,
        test_size=0.2,
        random_state=SEED,
    )
    train_index, val_index = [x for x in gss.split(df, groups=groups)][0]
    train_index = train_index[:TRAIN_SIZE]
    val_index = val_index[:VAL_SIZE]

    train_df: pd.DataFrame = df.iloc[train_index]
    val_df: pd.DataFrame = df.iloc[val_index]

    OUT_BASE_PATH.mkdir(exist_ok=True, parents=True)
    train_df.to_parquet(OUT_BASE_PATH / "train_mini.parquet")
    val_df.to_parquet(OUT_BASE_PATH / "val_mini.parquet")


if __name__ == '__main__':
    main()
