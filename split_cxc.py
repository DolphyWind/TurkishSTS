from pathlib import Path
from typing import List

from datasets import Dataset, load_dataset


def main():
    SEED: int = 42

    data_folder: Path = Path("./data/")
    data_files: List[str] = list(map(str, data_folder.glob("cxc-mt-turkish-*.jsonl")))
    cxc_dataset: Dataset = load_dataset("json", data_files=data_files)["train"]

    splits = cxc_dataset.train_test_split(test_size=0.2, seed=SEED)

    splits["train"].to_parquet(data_folder / "cxc-mt-turkish-80-20-split-train.parquet")
    splits["test"].to_parquet(data_folder / "cxc-mt-turkish-80-20-split-val.parquet")


if __name__ == '__main__':
    main()
