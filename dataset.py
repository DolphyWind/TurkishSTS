from pathlib import Path
from typing import Dict, List, Tuple
from datasets import DatasetDict, load_dataset, concatenate_datasets
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset


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
        # cache_dir="/run/media/yunus/HDD_5TB/hf_cache",
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
        elem: Dict[str, str] = self.hf_dataset[index]
        return elem


def simple_collate_fn(data: List[Dict[str, str]]) -> Tuple[List[str]]:
    batch: List[str] = []
    for anchor in data:
        batch.append(anchor["sentence"])
    return (batch,)
