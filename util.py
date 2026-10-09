from contextlib import contextmanager
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset


@contextmanager
def dropout_set_to(
    model: nn.Module,
    dropout_state: bool,
):
    mods = [mod for mod in model.modules() if isinstance(mod, nn.Dropout)]
    was_training = [mod.training for mod in mods]
    for mod in mods:
        mod.train(dropout_state)
    try:
        yield
    finally:
        for mod, flag in zip(mods, was_training):
            mod.train(flag)


def randn_like(
    x: torch.Tensor,
    mean: float = 0.0,
    var: float = 0.0,
) -> torch.Tensor:
    std: float = var ** .5
    g = torch.randn_like(x)
    return (mean + g * std)


def pick_subset(
    dataset: Dataset,
    data_size: float,
    generator: Optional[np.random.Generator] = None,
) -> Subset:
    if data_size < 0.0 or data_size > len(dataset):
        raise RuntimeError("Invalid data size!")
    if data_size > 1.0:
        data_size = int(data_size)
    else:
        data_size = int(len(dataset) * data_size)

    if generator is None:
        generator = np.random.default_rng()

    idx: npt.NDArray = np.arange(0, len(dataset))
    generator.shuffle(idx)
    idx = idx[:data_size]

    return Subset(dataset, idx)
