from contextlib import contextmanager
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def infonce_loss(
    *,
    anchors: torch.Tensor,
    positives: torch.Tensor,
    temperature: float,
    **kwargs,
) -> torch.Tensor:
    a = F.normalize(
        anchors,
        dim=1,
    )
    p = F.normalize(
        positives,
        dim=1,
    )

    logits: torch.Tensor = (a @ p.T).float() / temperature
    targets: torch.Tensor = torch.arange(
        logits.size(0),
        device=logits.device,
    )
    return F.cross_entropy(logits, targets)


def off_dropout_infonce_loss(
    *,
    anchors: torch.Tensor,
    positives: torch.Tensor,
    off_dropout: torch.Tensor,
    temperature: float,
    m: float = 0.9,
    **kwargs,
) -> torch.Tensor:
    a = F.normalize(anchors, dim=1)
    p = F.normalize(positives, dim=1)
    z = F.normalize(off_dropout, dim=1)

    n = a.size(0)

    pos = (a * p).sum(dim=1).float() / temperature

    neg = (z @ z.T).float() / temperature
    eye = torch.eye(n, dtype=torch.bool, device=neg.device)
    neg = neg.masked_fill(eye, float("-inf"))

    neg = neg + math.log(m)

    logits = torch.cat([pos.unsqueeze(1), neg], dim=1)
    targets = torch.zeros(n, dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, targets)


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
