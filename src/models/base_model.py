from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn import Module

from src.datasets.base_dataset import Batch


@dataclass
class ModelOutput:
    logits: torch.Tensor


@dataclass
class Loss:
    loss: torch.Tensor
    metrics: Optional[dict[str, float]]


class BaseModel(Module, ABC):
    @abstractmethod
    def forward(self, batch: Batch) -> ModelOutput:
        pass

    @abstractmethod
    def compute_loss(self, outputs: ModelOutput, batch: Batch) -> Loss:
        pass
