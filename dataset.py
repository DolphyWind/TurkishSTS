from pathlib import Path
from typing import Dict, List
from datasets import DatasetDict, load_dataset, concatenate_datasets
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset


BASE_PATH: Path = Path("/run/media/yunus/HDD_5TB/data/NLP/vngrs-web-corpus-sentences/")


def load_and_split(
    files: List[Path],
) -> HFDataset:

    segments: Dict[str, str] = {
        fname.stem.replace("-", "_"): fname.__str__()
        for fname in files
    }
    dataset: DatasetDict = load_dataset(
        'parquet',
        data_files=segments,
        cache_dir="/run/media/yunus/HDD_5TB/hf_cache",
        num_proc=8,
    )
    return concatenate_datasets(list(dataset.values()))


class SentenceDataset(Dataset):
    def __init__(
        self,
        hf_dataset: HFDataset,
    ) -> None:
        super().__init__()
        self.hf_dataset: HFDataset = hf_dataset

    def __len__(self) -> int:
        return self.hf_dataset.__len__()

    def __getitem__(self, index: int) -> Dict[str, str]:
        return self.hf_dataset[index]


def main():
    dataset = load_and_split(list(BASE_PATH.glob("*.parquet")))
    print(dataset[0:10])


if __name__ == '__main__':
    main()
