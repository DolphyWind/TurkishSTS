from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Tuple

from transformers import HfArgumentParser


@dataclass
class TrainConfig:
    batch_size: int = field(
        default=32,
        metadata={"help": "Batch size."},
    )
    num_epochs: int = field(
        default=20,
        metadata={"help": "Number of epochs."},
    )
    lr_peak: float = field(
        default=1e-5,
        metadata={"help": "Peak learning rate value. Default of AdamW if learning rate scheduler is disabled."},
    )
    lr_begin_factor: float = field(
        default=0.1,
        metadata={"help": "Linear warmup cosine decay begin factor"},
    )
    lr_end_factor: float = field(
        default=0,
        metadata={"help": "Linear warmup cosine decay end factor."},
    )
    weight_decay: float = field(
        default=1e-3,
        metadata={"help": "Weight decay parameter of AdamW."},
    )
    lr_scheduler: Literal["linear_warmup_cos_decay", "none"] = field(
        default="linear_warmup_cos_decay",
        metadata={"help": "Learning rate scheduler to use"},
    )
    warmup_percentage: float = field(
        default=0.1,
        metadata={"help": "Percentage of training steps for warming the learning rate up."},
    )
    mixed_precision: Literal["no", "fp16", "bf16"] = field(
        default="bf16",
        metadata={"help": "Mixed precision setting."},
    )
    infonce_temp: float = field(
        default=5e-2,
        metadata={"help": "InfoNCE Temperature."},
    )
    tradeoff_ratio: float = field(
        default=0.9,
        metadata={"help": "Trade-off ratio of Dimension-Wise Contrastive Learning. Denoted with `m` in the paper."},
    )
    dcl_temp: float = field(
        default=5.0,
        metadata={"help": "Temperature value of Dimension-Wise Contrastive Learning."},
    )
    dcl_coefficient: float = field(
        default=0.1,
        metadata={"help": "Loss coefficient of Dimension-Wise Contrastive Learning."},
    )
    noise_mean: float = field(
        default=0.0,
        metadata={"help": "Gaussian noise mean for the positive pairs."},
    )
    noise_var: float = field(
        default=0.1,
        metadata={"help": "Gaussian noise variance for the positive pairs."},
    )
    seed: Optional[int] = field(
        default=None,
        metadata={"help": "Seed value for reproducability."},
    )
    deterministic: bool = field(
        default=False,
        metadata={"help": "Use deterministic CUDA operations."},
    )


@dataclass
class DataConfig:
    pq_files: List[str] = field(
        default_factory=list,
        metadata={"help": "Paths to parquet files."},
    )
    val_pq_files: List[str] = field(
        default_factory=list,
        metadata={"help": "Parquet files for validation. Validation is optional but highly encouraged."},
    )
    sampler_to_use: Literal["regular", "near_far"] = field(
        default="near_far",
        metadata={"help": "Sampler to use."},
    )
    min_p: float = field(
        default=0.2,
        metadata={"help": "Minimum probability of selecting faraway samples (for NearFarSampler)"},
    )
    max_p: float = field(
        default=0.9,
        metadata={"help": "Maximum probability of selecting faraway samples (for NearFarSampler)"},
    )

    def __post_init__(self) -> None:
        if not self.pq_files:
            raise RuntimeError("Need at least one parquet file in order to train")
        self.pq_files = [Path(p) for p in self.pq_files]  # type: ignore[assignment]
        self.val_pq_files = [Path(p) for p in self.val_pq_files]  # type: ignore[assignment]


@dataclass
class ModelConfig:
    base_model: str
    initialize_new: bool = field(
        default=False,
        metadata={"help": "Initialize a new model rather than loading pretrained weights."},
    )
    lora: bool = field(
        default=False,
        metadata={"help": "Use LoRA for fine-tuning."},
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "Rank of LoRA matrics."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha value."},
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["query", "key", "value", "dense"],
        metadata={"help": "LoRA target modules."},
    )
    lora_modules_to_save: List[str] = field(
        default_factory=lambda: ["LayerNorm"],
        metadata={"help": "LoRA modules to save argument."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout."},
    )


@dataclass
class MonitoringConfig:
    experiment_name: str = field(
        metadata={"help": "Name of the MLFlow experiment."},
    )
    run_name: str = field(
        metadata={"help": "Name of the MLFlow run."},
    )
    resume_from: Optional[str] = field(
        default=None,
        metadata={"help": "Path to resume from."},
    )
    output_base_path: str = field(
        default="output/",
        metadata={"help": "Base output path."},
    )


def parse_args() -> Tuple[TrainConfig, DataConfig, ModelConfig, MonitoringConfig]:
    parser: HfArgumentParser = HfArgumentParser((
        TrainConfig,
        DataConfig,
        ModelConfig,
        MonitoringConfig,
    ))
    return parser.parse_args_into_dataclasses()
