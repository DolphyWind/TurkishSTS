from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset


def main():
    nli_dataset = load_dataset("emrecan/all-nli-tr", "pair-score")
    cxc_dataset = load_dataset("DolphyWind/cxc-sts-mt-turkish", "80-20-split")
    stsb_dataset = load_dataset("trmteb/stsb-tr")

    cxc_dataset = cxc_dataset.rename_columns({
        "caption1": "sentence1",
        "caption2": "sentence2",
        "agg_score": "score",
    })
    cxc_dataset = cxc_dataset.remove_columns("sampling_method")
    cxc_dataset = cxc_dataset.map(
        lambda batch: {"score": [s / 5.0 for s in batch["score"]]},
        batched=True,
    )
    stsb_dataset = stsb_dataset.map(
        lambda batch: {"score": [s / 5.0 for s in batch["score"]]},
        batched=True,
    )

    nli_dataset = nli_dataset.map(
        lambda batch: {"original_source": ["all-nli-tr"] * len(batch["score"])},
        batched=True,
    )
    cxc_dataset = cxc_dataset.map(
        lambda batch: {"original_source": ["cxc-sts-mt-turkish"] * len(batch["score"])},
        batched=True,
    )
    stsb_dataset = stsb_dataset.map(
        lambda batch: {"original_source": ["stsb-tr"] * len(batch["score"])},
        batched=True,
    )

    mixed: DatasetDict = DatasetDict({
        "train": concatenate_datasets([
            nli_dataset["train"],
            cxc_dataset["train"],
            stsb_dataset["train"],
        ]),
        "validation": concatenate_datasets([
            nli_dataset["dev"],
            cxc_dataset["validation"],
            stsb_dataset["validation"],
        ]),
        "test": concatenate_datasets([
            nli_dataset["test"],
            stsb_dataset["test"],
        ]),
    })
    mixed = mixed.filter(
        lambda row: row["sentence1"] is not None and row["sentence2"] is not None,
    )
    mixed = mixed.filter(
        lambda row: not (row["sentence1"] == row["sentence2"] and row["score"] != 1.0),
    )

    out_folder: Path = Path(__file__).parent.parent / "data"
    out_folder.mkdir(
        exist_ok=True,
        parents=True,
    )

    for split, dataset in mixed.items():
        split_dataset: Dataset = dataset
        split_dataset.to_parquet(str(out_folder / f"mixed-sts-{split}.parquet"))
        print(f"Written {split_dataset.__len__()} rows as {split} split.")


if __name__ == '__main__':
    main()
