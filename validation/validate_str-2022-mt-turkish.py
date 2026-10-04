from typing import List

from datasets import Dataset, DatasetDict, load_dataset

from validation_utils import evaluate_all_and_print


def main():
    dataset: DatasetDict = load_dataset(
        "json",
        data_files="data/str-2022-train.jsonl",
    )
    val: Dataset = dataset["train"]

    sentences1: List[str] = val["sentence1"]
    sentences2: List[str] = val["sentence2"]
    scores: List[float] = val["Score"]

    evaluate_all_and_print(
        sentences1=sentences1,
        sentences2=sentences2,
        scores=scores,
        max_model_len=25,
    )


if __name__ == '__main__':
    main()
