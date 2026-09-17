from dataclasses import dataclass, field
from typing import List, Tuple

from transformers import HFArgumentParser


@dataclass
class TrainConfig:
    pass


@dataclass
class DataConfig:
    pass


@dataclass
class ModelConfig:
    base_model: str
    initialize_new: bool = field(
        default=False,
        metadata={"help": "Initialize a new model rather than loading pretrained weights."},
    )
    lora: bool = field(
        default=False,
        metadata={"help": "Use LoRA for fine-tuning."}
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
        default_factory=list,
        metadata={"help": "LoRA target modules."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout."},
    )


def parse_args() -> Tuple[TrainConfig, DataConfig, ModelConfig]:
    parser: HFArgumentParser = HFArgumentParser((TrainConfig, DataConfig, ModelConfig))
    return parser.parse_into_data_classes()
