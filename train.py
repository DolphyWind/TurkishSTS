from contextlib import nullcontext
from dataclasses import asdict
import math
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, set_seed
from datasets import Dataset as HFDataset
import numpy as np
from peft import TaskType
from transformers import PreTrainedTokenizerBase
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import DataConfig, ModelConfig, MonitoringConfig, TrainConfig, parse_args
from dataset import SentenceDataset, TokenizingCollator, load_and_merge
from loss import dcl_loss, off_dropout_infonce_loss, infonce_loss_func
from lr_scheduler import LinearWarmupCosineDecay
from metrics import MetricLogger
from model import STSModel
from samplers import NearFarSampler
from util import dropout_set_to, pick_subset, randn_like


class Trainer:
    def __init__(
        self,
        train_config: TrainConfig,
        data_config: DataConfig,
        model_config: ModelConfig,
        monitoring_config: MonitoringConfig,
    ) -> None:
        self.train_config: TrainConfig = train_config
        self.data_config: DataConfig = data_config
        self.model_config: ModelConfig = model_config
        self.monitoring_config: MonitoringConfig = monitoring_config

        resume_path = self.monitoring_config.resume_from

        if resume_path is not None:
            config_path: Path = Path(resume_path) / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"{config_path} does not exist.")
            saved: Dict[str, Any] = json.loads(config_path.read_text())
            train_config = TrainConfig(**saved["train_config"])
            data_config = DataConfig(**saved["data_config"])
            model_config = ModelConfig(**saved["model_config"])
            monitoring_config = MonitoringConfig(**saved["monitoring_config"])
            monitoring_config.resume_from = resume_path

            self.train_config = train_config
            self.data_config = data_config
            self.model_config = model_config
            self.monitoring_config = monitoring_config

        self.accelerator = Accelerator(
            mixed_precision=train_config.mixed_precision,
            dataloader_config=DataLoaderConfiguration(
                use_seedable_sampler=True,
                use_stateful_dataloader=True,
            ),
        )

        self.device: torch.device = self.accelerator.device
        self.epoch: int = 1
        self.run_id: Optional[str] = None
        self.best_epoch: Optional[int] = None
        self.best_loss: float = float("inf")

        self.seed: Optional[int] = self.train_config.seed
        self.deterministic: bool = self.train_config.deterministic

        self._seed()

        self.model: STSModel = STSModel(
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
                "modules_to_save": model_config.lora_modules_to_save
            }
        )
        self.__hf_dataset: HFDataset = load_and_merge(data_config.pq_files)  # pyright: ignore
        self.__val_hf_dataset: Optional[HFDataset] = load_and_merge(
            data_config.val_pq_files  # pyright: ignore
        ) if data_config.val_pq_files else None

        self.train_dataset: Union[SentenceDataset, Subset] = SentenceDataset(
            hf_dataset=self.__hf_dataset,
        )
        self.train_dataset = pick_subset(
            self.train_dataset,
            data_size=self.data_config.train_size,
            generator=np.random.default_rng(self.seed and (self.seed, 0)),
        )

        self.val_dataset: Optional[Union[SentenceDataset, Subset]] = None
        if self.__val_hf_dataset:
            self.val_dataset = SentenceDataset(
                hf_dataset=self.__val_hf_dataset,
            )
            self.val_dataset = pick_subset(
                self.val_dataset,
                data_size=self.data_config.val_size,
                generator=np.random.default_rng(self.seed and (self.seed, 1)),
            )

        self.train_loader: DataLoader
        self.train_sampler: Optional[NearFarSampler]
        self._initialize_sampler_and_loader()

        self.val_loader: Optional[DataLoader] = None
        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=train_config.batch_size,
                shuffle=False,
                collate_fn=TokenizingCollator(self.model.tokenizer),
                num_workers=self.data_config.num_workers,
                pin_memory=True,
                drop_last=(len(self.val_dataset) % train_config.batch_size < 16),
            )

        self.optimizer: optim.Optimizer = optim.AdamW(
            self.model.parameters(),
            lr=train_config.lr_peak,
            weight_decay=train_config.weight_decay,
            fused=True,
        )
        self.lr_scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
        match train_config.lr_scheduler:
            case "linear_warmup_cos_decay":
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
        self.tokenizer: PreTrainedTokenizerBase = self.model.tokenizer

        if self._validation:
            self.model, self.optimizer, self.train_loader, self.val_loader, self.lr_scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.val_loader, self.lr_scheduler
            )
        else:
            self.model, self.optimizer, self.train_loader, self.lr_scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.lr_scheduler
            )
        self.accelerator.register_for_checkpointing(self)

        self.out_path: Path = Path(monitoring_config.output_base_path) / monitoring_config.experiment_name / monitoring_config.run_name
        self.out_path.mkdir(exist_ok=True, parents=True)
        if any(self.out_path.iterdir()) and resume_path is None:
            raise RuntimeError("The output directory exist and is non-empty.")

        self.accelerate_save_path: Path = self.out_path / "state"

        if resume_path:
            self.accelerator.load_state(str(Path(resume_path) / "state"))
        else:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                saved: Dict[str, Any] = {
                    "train_config": asdict(self.train_config),
                    "data_config": asdict(self.data_config),
                    "model_config": asdict(self.model_config),
                    "monitoring_config": asdict(self.monitoring_config),
                }
                (self.out_path / "config.json").write_text(
                    json.dumps(
                        saved,
                        indent=4,
                        default=str,
                    ),
                )

    def train(self) -> None:
        metric_logger = MetricLogger(
            is_main_process=self.accelerator.is_main_process,
            experiment_name=self.monitoring_config.experiment_name,
            run_name=self.monitoring_config.run_name,
            run_id=self.run_id,
            out_dir=self.out_path,
        )

        for cfg in (
            self.train_config,
            self.model_config,
            self.monitoring_config,
            self.data_config,
        ):
            metric_logger.log_cfg(cfg)

        with metric_logger.ctx as run:
            self.run_id = run.info.run_id if run else self.run_id
            metric_logger.set_run_id(self.run_id)

            print("Training started!")
            try:
                for epoch in range(self.epoch, self.train_config.num_epochs + 1):
                    # Train loss is never None. This is to avoid pyright errors
                    train_metrics: Dict[str, float] = self._one_step(epoch, 'train') or {"loss": float("inf")}
                    val_metrics: Optional[Dict[str, float]] = self._one_step(epoch, 'val')

                    for name, value in train_metrics.items():
                        metric_logger.log_metric(
                            f"train_{name}",
                            value,
                            step=epoch,
                        )
                    if val_metrics is not None:
                        for name, value in val_metrics.items():
                            metric_logger.log_metric(
                                f"val_{name}",
                                value,
                                step=epoch,
                            )

                    tracked_loss: float = val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
                    is_best: bool = tracked_loss < self.best_loss

                    if is_best:
                        self.best_epoch = epoch
                        self.best_loss = tracked_loss

                    self.accelerator.wait_for_everyone()
                    if is_best:
                        unwrapped = self.accelerator.unwrap_model(self.model)

                        if self.accelerator.is_main_process:
                            self.accelerator.save(
                                unwrapped.state_dict(),
                                self.out_path / "model_best.pth",
                            )

                    self.epoch = epoch + 1
                    match self.data_config.sampler_to_use:
                        case "near_far":
                            assert isinstance(self.train_sampler, NearFarSampler), "Stupid type system"
                            self.train_sampler.set_epoch(self.epoch)
                        case "regular":
                            pass
                    metric_logger.write_to_disk()
                    self.accelerator.save_state(
                        str(self.accelerate_save_path)
                    )
            finally:
                metric_logger.stop()

    def _initialize_sampler_and_loader(self) -> None:
        match self.data_config.sampler_to_use:
            case "near_far":
                self.train_sampler = NearFarSampler(
                    data_len=len(self.train_dataset),
                    batch_size=self.train_config.batch_size,
                    seed=self.seed or 0,
                    max_epochs=self.train_config.num_epochs,
                    min_p=self.data_config.min_p,
                    max_p=self.data_config.max_p,
                    drop_last=(len(self.train_dataset) % self.train_config.batch_size < 16),
                )
                self.train_loader = DataLoader(
                    self.train_dataset,
                    batch_sampler=self.train_sampler,
                    collate_fn=TokenizingCollator(self.model.tokenizer),
                    num_workers=self.data_config.num_workers,
                    pin_memory=True,
                )
            case "regular":
                self.train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=self.train_config.batch_size,
                    shuffle=True,
                    collate_fn=TokenizingCollator(self.model.tokenizer),
                    num_workers=self.data_config.num_workers,
                    pin_memory=True,
                    drop_last=(len(self.train_dataset) % self.train_config.batch_size < 16),
                )

    def _seed(self) -> None:
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
    ) -> Optional[Dict[str, float]]:
        if mode == 'val' and not self._validation:
            return None
        self.model.train(mode == 'train')

        loop: tqdm
        if mode == 'train':
            self.train_loader.set_epoch(epoch_num)  # pyright: ignore
            loop = tqdm(
                self.train_loader,
                disable=not self.accelerator.is_main_process,
            )
        elif mode == 'val':
            self.val_loader.set_epoch(epoch_num)  # pyright: ignore
            loop = tqdm(
                self.val_loader,
                disable=not self.accelerator.is_main_process,
            )

        desc: str = "Epoch" if mode == 'train' else "Validation"
        loop.set_description_str(f"{desc} [{epoch_num}/{self.train_config.num_epochs}]")
        total_loss: torch.Tensor = torch.zeros(
            (),
            device=self.device,
        )
        total_infonce: torch.Tensor = torch.zeros(
            (),
            device=self.device,
        )
        total_dcl: torch.Tensor = torch.zeros(
            (),
            device=self.device,
        )
        losses: List[torch.Tensor] = []
        infonce_losses: List[torch.Tensor] = []
        dcl_losses: List[torch.Tensor] = []

        grad_ctx = nullcontext() if mode == 'train' else torch.no_grad()
        rng_ctx = nullcontext()
        if mode == 'val':
            rng_ctx = torch.random.fork_rng(
                devices=[self.device] if self.device.type == "cuda" else [],
            )
        with grad_ctx, rng_ctx:
            if mode == 'val':
                torch.manual_seed(self.seed or 0)
            for i, anchors_pt in enumerate(loop, start=1):
                if mode == 'train':
                    self.optimizer.zero_grad()

                doubled_pt: Dict[str, torch.Tensor] = {
                    key: torch.cat([value, value])
                    for key, value in anchors_pt.items()
                }

                z1_orig: torch.Tensor
                z2_orig: torch.Tensor
                with dropout_set_to(self.model, True):
                    z1_orig, z2_orig = self.model(**doubled_pt).chunk(2)

                with dropout_set_to(self.model, False):
                    z_off: torch.Tensor = self.model(**anchors_pt)

                infonce_loss: torch.Tensor
                dcl: torch.Tensor
                loss: torch.Tensor

                if self.train_config.methodology == "simcse++":
                    z1: torch.Tensor = z1_orig + randn_like(
                        z1_orig,
                        mean=self.train_config.noise_mean,
                        var=self.train_config.noise_var,
                    )
                    z2: torch.Tensor = z2_orig + randn_like(
                        z2_orig,
                        mean=self.train_config.noise_mean,
                        var=self.train_config.noise_var,
                    )

                    infonce_loss = off_dropout_infonce_loss(
                        anchors=z1,
                        positives=z2,
                        off_dropout=z_off,
                        temperature=self.train_config.infonce_temp,
                        m=self.train_config.tradeoff_ratio,
                    )
                elif self.train_config.methodology == "simcse":
                    infonce_loss = infonce_loss_func(
                        anchors=z1_orig,
                        positives=z2_orig,
                        temperature=self.train_config.infonce_temp,
                    )
                else:
                    # Gets rid of pyright's possibly unbound errors
                    raise RuntimeError("Unreachable.")

                dcl = dcl_loss(
                    anchors=z1_orig,
                    positives=z2_orig,
                    temperature=self.train_config.dcl_temp,
                )
                loss = infonce_loss + self.train_config.dcl_coefficient * dcl

                if mode == 'train':
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    if self.lr_scheduler:
                        self.lr_scheduler.step()

                loss_detached: torch.Tensor = loss.detach()
                infonce_detached: torch.Tensor = infonce_loss.detach()
                dcl_detached: torch.Tensor = dcl.detach() * self.train_config.dcl_coefficient
                total_loss += loss_detached
                total_infonce += infonce_detached
                total_dcl += dcl_detached

                losses.append(loss_detached)
                infonce_losses.append(infonce_detached)
                dcl_losses.append(dcl_detached)

                if i % self.train_config.log_interval == 0:
                    synced: List[float] = torch.stack([
                        loss_detached,
                        infonce_detached,
                        dcl_detached,
                        total_loss,
                        total_infonce,
                        total_dcl,
                    ]).tolist()
                    lr: float = self.optimizer.param_groups[0]["lr"]
                    loop.set_postfix(
                        loss=synced[0],
                        infonce=synced[1],
                        dcl=synced[2],
                        avg_loss=synced[3] / i,
                        avg_infonce=synced[4] / i,
                        avg_dcl=synced[5] / i,
                        lr=lr,
                    )

        metrics: Dict[str, float] = {}
        for name, values in (
            ("loss", losses),
            ("infonce", infonce_losses),
            ("dcl", dcl_losses),
        ):
            values_tensor: torch.Tensor = torch.stack(values)
            values_gathered: torch.Tensor = self.accelerator.gather_for_metrics(values_tensor)
            metrics[name] = values_gathered.mean().item()

        return metrics

    def state_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "epoch": self.epoch,
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
            "train_config": asdict(self.train_config),
            "data_config": asdict(self.data_config),
            "model_config": asdict(self.model_config),
            "monitoring_config": asdict(self.monitoring_config),
        }

    def load_state_dict(
        self,
        state: Dict[str, Any],
    ) -> None:
        self.run_id = state.get("run_id", None)
        self.epoch = state.get("epoch", 1)
        self.best_epoch = state.get("best_epoch", None)
        self.best_loss = state.get("best_loss", float("inf"))

        self.train_config = TrainConfig(**state["train_config"])
        self.data_config = DataConfig(**state["data_config"])
        self.model_config = ModelConfig(**state["model_config"])
        self.monitoring_config = MonitoringConfig(**state["monitoring_config"])


def main():
    train_config, data_config, model_config, monitoring_config = parse_args()
    trainer = Trainer(
        train_config,
        data_config,
        model_config,
        monitoring_config,
    )
    trainer.train()


if __name__ == '__main__':
    main()
