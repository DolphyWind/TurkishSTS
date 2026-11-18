import os
from pathlib import Path
import sys
from typing import Optional, Tuple, Union

from datasets import Dataset, DatasetDict, load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.sentence_transformer.training_args import BatchSamplers

from util import load_model


def load_datasets() -> Tuple[Dataset, Dataset]:
    data_folder: Path = Path(__file__).parent / "data"
    dataset: DatasetDict = load_dataset(
        "parquet",
        data_files={
            "train": str(data_folder / "web-pairs-train.parquet"),
            "validation": str(data_folder / "web-pairs-validation.parquet"),
        },
    )
    dataset = dataset.remove_columns("source")
    return dataset["train"], dataset["validation"]


def main():
    experiment_name: Optional[str] = os.environ.get("MLFLOW_EXPERIMENT_NAME", None)
    if experiment_name is None:
        print("Please set 'MLFLOW_EXPERIMENT_NAME' variable!", file=sys.stderr)
        exit(1)

    model_name: Union[str, Path] = ""
    model: SentenceTransformer = load_model(model_name)

    train_dataset: Dataset
    eval_dataset: Dataset
    train_dataset, eval_dataset = load_datasets()

    train_loss = losses.CachedMultipleNegativesRankingLoss(
        model=model,
        mini_batch_size=128,
    )

    num_epochs: int = 10
    run_name: str = "web_pairs"

    args = SentenceTransformerTrainingArguments(
        output_dir=f"output/{experiment_name}/{run_name}",
        run_name=run_name,
        per_device_train_batch_size=1024,
        per_device_eval_batch_size=256,
        num_train_epochs=num_epochs,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,
        weight_decay=1e-3,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch",
        eval_on_start=True,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=100,
        report_to="mlflow",
        dataloader_num_workers=4,
    )

    trainer: SentenceTransformerTrainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
    )
    trainer.train()


if __name__ == '__main__':
    main()
