from contextlib import nullcontext
import random
from typing import Literal, Optional

from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, set_seed
from datasets import Dataset as HFDataset
import numpy as np
from peft import TaskType
from transformers import PreTrainedTokenizerBase
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DataConfig, ModelConfig, TrainConfig, parse_args
from dataset import SentenceDataset, load_and_merge, simple_collate_fn
from loss import dcl_loss, off_dropout_infonce_loss
from lr_scheduler import LinearWarmupCosineDecay
from model import STSModel
from util import dropout_set_to, randn_like


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

        self.accelerator = Accelerator(
            mixed_precision=train_config.mixed_precision,
            dataloader_config=DataLoaderConfiguration(
                use_seedable_sampler=True,
                use_stateful_dataloader=True,
            ),
        )
        self.device: torch.device = self.accelerator.device

        # TODO: Check if there is anything to resume from. If so handle it here
        self.seed: Optional[int] = self.train_config.seed
        self.deterministic: bool = self.train_config.deterministic

        self._seed()

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

        self._validation: bool = self.val_loader is not None
        self.num_epochs: int = train_config.num_epochs
        self.tokenizer: PreTrainedTokenizerBase = self.model.tokenizer

        if self._validation:
            self.model, self.optimizer, self.train_loader, self.val_loader, self.lr_scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.val_loader, self.lr_scheduler
            )
        else:
            self.model, self.optimizer, self.train_loader, self.lr_scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.lr_scheduler
            )

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            current_loss: float = self._one_step(epoch, 'train') or float('inf')
            current_loss = self._one_step(epoch, 'val') or current_loss

    def _seed(self):
        if self.seed is None:
            return

        set_seed(
            self.seed,
            deterministic=self.deterministic,
        )
        if self.deterministic:
            torch.backends.cudnn.benchmark = False

    def _one_step(
        self,
        epoch_num: int,
        mode: Literal['train', 'val'],
    ) -> Optional[float]:
        if mode == 'val' and not self._validation:
            return None
        self.model.train(mode == 'train')

        loop: tqdm
        if mode == 'train':
            self.train_loader.set_epoch(epoch_num)  # pyright: ignore
            loop = tqdm(self.train_loader)
        elif mode == 'val':
            self.val_loader.set_epoch(epoch_num)  # pyright: ignore
            loop = tqdm(self.val_loader)

        desc: str = "Epoch" if mode == 'train' else "Validation"
        loop.set_description_str(f"{desc} [{epoch_num}/{self.num_epochs}]")
        total_loss: float = 0.0
        avg_loss: float = 0.0
        num_elems: int = len(loop)

        ctx = nullcontext() if mode == 'train' else torch.no_grad()
        with ctx:
            for i, (anchors,) in enumerate(loop, start=1):
                anchors_pt = self.tokenizer(
                    anchors,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )

                anchors_pt = anchors_pt.to(self.device)

                if mode == 'train':
                    self.optimizer.zero_grad()

                with dropout_set_to(self.model, True):
                    z1: torch.Tensor = self.model(**anchors_pt)
                    z2: torch.Tensor = self.model(**anchors_pt)

                with dropout_set_to(self.model, False):
                    z_off: torch.Tensor = self.model(**anchors_pt)

                z1 = z1 + randn_like(
                    z1,
                    mean=self.train_config.noise_mean,
                    var=self.train_config.noise_var,
                )
                z2 = z2 + randn_like(
                    z2,
                    mean=self.train_config.noise_mean,
                    var=self.train_config.noise_var,
                )

                infonce_loss: torch.Tensor = off_dropout_infonce_loss(
                    anchors=z1,
                    positives=z2,
                    off_dropout=z_off,
                    temperature=self.train_config.infonce_temp,
                    m=self.train_config.tradeoff_ratio,
                )
                dcl: torch.Tensor = dcl_loss(
                    anchors=z1,
                    positives=z2,
                    temperature=self.train_config.dcl_temp,
                )
                loss = infonce_loss + self.train_config.dcl_coefficient * dcl

                if mode == 'train':
                    loss.backward()
                    self.optimizer.step()
                    if self.lr_scheduler:
                        self.lr_scheduler.step()

                loss_float: float = loss.item()
                total_loss += loss_float
                avg_loss = total_loss / i
                lr: float = self.optimizer.param_groups[0]["lr"]
                loop.set_postfix(
                    loss=loss_float,
                    infonce=infonce_loss.item(),
                    dcl=dcl.item() * self.train_config.dcl_coefficient,
                    avg_loss=avg_loss,
                    lr=lr,
                )

        return total_loss / num_elems


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
