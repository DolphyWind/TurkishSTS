from contextlib import contextmanager
from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch.utils.data import Subset

from dataset import SentenceDataset


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
