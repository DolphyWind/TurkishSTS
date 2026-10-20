from typing import List

from datasets import Dataset, DatasetDict, load_dataset

from validation_utils import evaluate_all_and_print


def main():
    dataset: DatasetDict = load_dataset("DolphyWind/cxc-sts-mt-turkish", "80-20-split")
    val: Dataset = dataset["validation"]

    sentences1: List[str] = val["caption1"]
    sentences2: List[str] = val["caption2"]
    scores: List[float] = [s / 5.0 for s in val["agg_score"]]

    evaluate_all_and_print(
        sentences1=sentences1,
        sentences2=sentences2,
        scores=scores,
        max_model_len=25,
        dataset_name="cxc-mt-turkish",
    )


if __name__ == '__main__':
    main()
