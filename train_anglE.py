import json
from pathlib import Path
import tempfile
from typing import Any, Dict, Union

from peft import PeftModel, TaskType
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.sentence_transformer.modules import Pooling, Transformer
from datasets import Dataset
import torch

from config import ModelConfig
from model import STSModel


def load_model(model_name: Union[str, Path]) -> SentenceTransformer:
    if isinstance(model_name, str):
        return SentenceTransformer(model_name)

    config_path: Path = model_name.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} does not exist.")
    saved: Dict[str, Any] = json.loads(config_path.read_text())
    model_config: ModelConfig = ModelConfig(**saved["model_config"])
    sts_model: STSModel = STSModel(
        base_model_name=model_config.base_model,
        initialize_new=model_config.initialize_new,
        lora=model_config.lora,
        lora_config={
            "r": model_config.lora_r,
            "lora_alpha": model_config.lora_alpha,
            "lora_dropout": model_config.lora_dropout,
            "bias": "none",
            "task_type": TaskType.FEATURE_EXTRACTION,
            "target_modules": model_config.lora_target_modules,
            "modules_to_save": model_config.lora_modules_to_save,
        },
    )
    state_dict: Dict[str, torch.Tensor] = torch.load(
        model_name,
        map_location="cpu",
    )
    sts_model.load_state_dict(state_dict)
    if isinstance(sts_model.base, PeftModel):
        sts_model.base = sts_model.base.merge_and_unload()

    with tempfile.TemporaryDirectory() as tmp_dir:
        sts_model.base.save_pretrained(tmp_dir)
        sts_model.tokenizer.save_pretrained(tmp_dir)
        transformer: Transformer = Transformer(tmp_dir)
        pooling: Pooling = Pooling(
            transformer.get_embedding_dimension(),
            "mean",
        )
        return SentenceTransformer(modules=[transformer, pooling])


def main():
    model_name: Union[str, Path] = ""
    model: SentenceTransformer = load_model(model_name)
    dataset = None  # Implement later
    eval_dataset = None  # Implement later

    train_loss = losses.AnglELoss(model=model)

    num_epochs: int = 20
    run_name: str = "anglE"

    args = SentenceTransformerTrainingArguments(
        output_dir="output/anglE/",
        run_name=run_name,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        num_train_epochs=num_epochs,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_steps=1.0 / num_epochs,
        weight_decay=1e-3,
        bf16=True,
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
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
    )
    trainer.train()


if __name__ == '__main__':
    main()
