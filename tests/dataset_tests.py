from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader

from dataset import SentenceDataset, load_and_merge, simple_collate_fn


BASE_PATH: Path = Path("/run/media/yunus/HDD_5TB/data/NLP/vngrs-web-corpus-sentences/")


def main():
    pq_files = list(BASE_PATH.glob("*.parquet"))
    hf_dataset = load_and_merge([pq_files[0]])
    sentence_dataset = SentenceDataset(hf_dataset)
    loader = DataLoader(
        sentence_dataset,
        batch_size=5,
        shuffle=True,
        collate_fn=simple_collate_fn,
    )
    for x, y in loader:
        print(x)
        print(y)
        break


if __name__ == '__main__':
    main()
