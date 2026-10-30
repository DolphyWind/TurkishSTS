from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
from peft import PeftModel, TaskType
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Pooling, Transformer
import torch
import torch.nn as nn
from torch.utils.data import Subset

from config import ModelConfig
from dataset import SentenceDataset
from model import STSModel


@contextmanager
def dropout_set_to(
    model: nn.Module,
    dropout_state: bool,
):
    # Set training=False for all modules because some "geniuses" decided to use nn.functional.dropout
    mods: List[nn.Module] = [mod for mod in model.modules()]
    was_training: List[bool] = [mod.training for mod in mods]
    for mod in mods:
        mod.training = dropout_state
    try:
        yield
    finally:
        for mod, flag in zip(mods, was_training):
            mod.training = flag


def randn_like(
    x: torch.Tensor,
    mean: float = 0.0,
    var: float = 0.0,
) -> torch.Tensor:
    std: float = var ** .5
    g = torch.randn_like(x)
    return (mean + g * std)


def pick_subset(
    dataset: SentenceDataset,
    data_size: float,
    generator: Optional[np.random.Generator] = None,
) -> Subset:
    if data_size < 0.0 or data_size > len(dataset):
        raise RuntimeError("Invalid data size!")
    if data_size == 1.0:
        return Subset(dataset, range(len(dataset)))
    if data_size > 1.0:
        data_size = int(data_size)
    else:
        data_size = int(len(dataset) * data_size)

    if generator is None:
        generator = np.random.default_rng()

    sources: npt.NDArray = np.asarray(dataset.hf_dataset["source"])
    group_ids: npt.NDArray
    _, group_ids = np.unique(
        sources,
        return_inverse=True,
    )
    group_rank: npt.NDArray = generator.permutation(group_ids.max() + 1)
    idx: npt.NDArray = np.argsort(
        group_rank[group_ids],
        kind="stable",
    )
    idx = idx[:data_size]

    return Subset(dataset, idx)


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
