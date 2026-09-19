from typing import Optional

from datasets import Dataset as HFDataset
from peft import TaskType
import torch.optim as optim
from torch.utils.data import DataLoader

from config import DataConfig, ModelConfig, TrainConfig, parse_args
from dataset import SentenceDataset, load_and_merge, simple_collate_fn
from lr_scheduler import LinearWarmupCosineDecay
from model import STSModel


class Trainer:
    def __init__(
        self,
        train_config: TrainConfig,
        data_config: DataConfig,
        model_config: ModelConfig,
    ) -> None:
        self.train_config: TrainConfig = train_config
        self.data_config: DataConfig = data_config
        self.model_config: ModelConfig = model_config

        # TODO: Check if there is anything to resume from. If so handle it here

        self.model: STSModel = STSModel(
            base_model=model_config.base_model,
            initialize_new=model_config.initialize_new,
            lora=model_config.lora,
            lora_config={
                "r": model_config.lora_r,
                "lora_alpha": model_config.lora_alpha,
                "lora_dropout": model_config.lora_dropout,
                "bias": "none",
                "task_type": TaskType.FEATURE_EXTRACTION,
                "target_modules": model_config.lora_target_modules,
                "modules_to_save": model_config.lora_modules_to_save
            }
        )
        self.__hf_dataset: HFDataset = load_and_merge(data_config.pq_files)  # pyright: ignore
        self.__val_hf_dataset: Optional[HFDataset] = load_and_merge(
            data_config.val_pq_files  # pyright: ignore
        ) if data_config.val_pq_files else None
        self.train_dataset: SentenceDataset = SentenceDataset(
            hf_dataset=self.__hf_dataset,
        )
        self.val_dataset: Optional[SentenceDataset] = SentenceDataset(
            hf_dataset=self.__val_hf_dataset,
        ) if self.__val_hf_dataset else None

        self.train_loader: DataLoader = DataLoader(
            self.train_dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            collate_fn=simple_collate_fn,
        )
        self.val_loader: Optional[DataLoader] = DataLoader(
            self.val_dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            collate_fn=simple_collate_fn,
        ) if self.val_dataset else None

        self.optimizer: optim.Optimizer = optim.AdamW(
            self.model.parameters(),
            lr=train_config.lr_peak,
            weight_decay=train_config.weight_decay,
        )
        self.lr_scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
        match train_config.lr_scheduler:
            case "linear_warmup_cos_decay":
                import math
                total_steps: int = math.ceil(
                    self.train_dataset.__len__() / train_config.batch_size
                ) * train_config.num_epochs
                warmup_steps: int = int(total_steps * train_config.warmup_percentage)
                self.lr_scheduler = LinearWarmupCosineDecay(
                    self.optimizer,
                    start_factor=train_config.lr_begin_factor,
                    end_factor=train_config.lr_end_factor,
                    warmup_steps=warmup_steps,
                    total_steps=total_steps,
                    last_epoch=-1,
                )
            case "none":
                pass

    def train(self):
        pass


def main():
    train_config, data_config, model_config = parse_args()
    trainer = Trainer(
        train_config,
        data_config,
        model_config
    )
    trainer.train()


if __name__ == '__main__':
    main()
