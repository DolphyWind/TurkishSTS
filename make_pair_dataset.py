# This script creates a pair dataset from the web corpus
# Sentences coming from the same source always seem similar so idrk
# how effective this dataset will be
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

from datasets import Dataset, DatasetDict
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm


CORPUS_DIR: Path = Path("/run/media/yunus/HDD_5TB/data/NLP/vngrs-web-corpus-sentences")
OBFUSCATED_EMAIL: str = "[email protected]"


def iter_rows(path: Path) -> Iterator[Tuple[Optional[str], str]]:
    parquet_file: pq.ParquetFile = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(columns=["sentence", "source"]):
        yield from zip(
            batch.column("sentence").to_pylist(),
            batch.column("source").to_pylist(),
        )


def is_valid(
    sentence: Optional[str],
    seen: Set[int],
    min_words: int,
) -> bool:
    if sentence is None:
        return False
    if len(sentence.split()) < min_words or OBFUSCATED_EMAIL in sentence:
        return False
    key: int = hash(sentence)
    if key in seen:
        return False
    seen.add(key)
    return True


def make_pairs(
    shards: List[Path],
    target_pairs: int,
    min_words: int,
) -> Dict[str, List[str]]:
    pairs: Dict[str, List[str]] = {
        "sentence1": [],
        "sentence2": [],
        "source": [],
    }
    seen: Set[int] = set()
    prev_sentence: Optional[str] = None
    prev_source: Optional[str] = None
    progress: tqdm = tqdm(total=target_pairs)

    for shard in shards:
        for sentence, source in iter_rows(shard):
            valid: bool = is_valid(
                sentence,
                seen,
                min_words,
            )
            if valid and prev_sentence is not None and prev_source == source:
                pairs["sentence1"].append(prev_sentence)
                pairs["sentence2"].append(sentence)
                pairs["source"].append(source)
                prev_sentence = None
                progress.update(1)
                if len(pairs["sentence1"]) >= target_pairs:
                    progress.close()
                    return pairs
            else:
                prev_sentence = sentence if valid else None
                prev_source = source

    progress.close()
    return pairs


def main() -> None:
    target_pairs: int = 1_000_000
    min_words: int = 5
    val_size: int = 10_000
    split_seed: int = 42

    shards: List[Path] = sorted(CORPUS_DIR.glob("*.parquet"))
    pairs: Dict[str, List[str]] = make_pairs(
        shards,
        target_pairs,
        min_words,
    )

    dataset: Dataset = Dataset.from_dict(pairs)
    splits: DatasetDict = dataset.train_test_split(
        test_size=val_size,
        generator=np.random.default_rng(split_seed),
    )

    out_folder: Path = Path(__file__).parent / "data"
    out_folder.mkdir(
        exist_ok=True,
        parents=True,
    )

    for split, split_dataset in (("train", splits["train"]), ("validation", splits["test"])):
        split_dataset.to_parquet(str(out_folder / f"web-pairs-{split}.parquet"))
        print(f"Written {len(split_dataset)} rows as {split} split.")


if __name__ == '__main__':
    main()
