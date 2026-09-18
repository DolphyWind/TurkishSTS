from pathlib import Path
from typing import Dict, List, Tuple
from datasets import DatasetDict, load_dataset, concatenate_datasets
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset


BASE_PATH: Path = Path("/run/media/yunus/HDD_5TB/data/NLP/vngrs-web-corpus-sentences/")


def load_and_merge(
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

    def __getitem__(self, index: int) -> Tuple[str, str]:
        sentence: str = self.hf_dataset[index]["sentence"]
        # I'll add separate data augmentations later
        return sentence, sentence


def simple_collate_fn(data: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
    anchors: List[str] = []
    positives: List[str] = []
    for anchor, positive in data:
        anchors.append(anchor)
        positives.append(positive)
    return anchors, positives


def main():
    dataset = load_and_merge(list(BASE_PATH.glob("*.parquet")))
    print(dataset[0:10])


if __name__ == '__main__':
    main()
