from contextlib import contextmanager

import torch
import torch.nn as nn


@contextmanager
def dropout_disabled(model: nn.Module):
    mods = [mod for mod in model.modules() if isinstance(mod, nn.Dropout)]
    was_training = [mod.training for mod in mods]
    for mod in mods:
        mod.eval()
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
