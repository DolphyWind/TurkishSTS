import math
from typing import Any, Dict, Iterator, List

import numpy as np
import numpy.typing as npt
from torch.utils.data import Sampler


class NearFarSampler(Sampler[List[int]]):
    def __init__(
        self,
        data_len: int,
        batch_size: int,
        seed: int,
        max_epochs: int,
        min_p: float = 0.2,
        max_p: float = 0.9,
        drop_last: bool = False,
    ) -> None:
        super().__init__()

        self.data_len: int = data_len
        self.batch_size: int = batch_size
        self.drop_last: bool = drop_last
        self.min_p: float = min_p
        self.max_p: float = max_p
        self.seed: int = seed
        self.epoch: int = 1
        self.max_epochs: int = max_epochs

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def get_p(self) -> float:
        x: float = (self.epoch - 1) / (self.max_epochs - 1)
        alpha: float = (math.cos(math.pi * x) + 1.0) * 0.5
        return self.min_p * (1 - alpha) + self.max_p * alpha

    def __len__(self) -> int:
        if self.drop_last:
            return self.data_len // self.batch_size
        return math.ceil(self.data_len / self.batch_size)

    def __iter__(self) -> Iterator[List[int]]:
        gen: np.random.Generator = np.random.default_rng((self.seed, self.epoch))
        idx: npt.NDArray = np.arange(self.data_len)
        shift = gen.integers(self.data_len)
        idx = np.roll(idx, shift)
        p: float = self.get_p()

        pos: npt.NDArray = np.flatnonzero(gen.random(self.data_len) < p)
        idx[pos] = idx[gen.permutation(pos)]

        flat: List[int] = idx.tolist()
        for i in range(len(self)):
            yield flat[(i * self.batch_size):(i + 1) * self.batch_size]

    def load_state_dict(self, state: Dict[str, Any]):
        self.min_p = state["min_p"]
        self.max_p = state["max_p"]
        self.data_len = state["data_len"]
        self.batch_size = state["batch_size"]
        self.drop_last = state["drop_last"]
        self.seed = state["seed"]
        # self.epoch = state["epoch"]
        self.max_epochs = state["max_epochs"]

    def state_dict(self) -> Dict[str, Any]:
        return {
            "min_p": self.min_p,
            "max_p": self.max_p,
            "data_len": self.data_len,
            "batch_size": self.batch_size,
            "drop_last": self.drop_last,
            "seed": self.seed,
            "epoch": self.epoch,
            "max_epochs": self.max_epochs,
        }
