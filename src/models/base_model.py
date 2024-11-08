from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Generic, TypeVar

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


B = TypeVar("B", bound=Batch)


class BaseModel(Module, ABC, Generic[B]):
    @abstractmethod
    def forward(self, batch: B) -> ModelOutput:
        pass

    @abstractmethod
    def compute_loss(self, outputs: ModelOutput, batch: B) -> Loss:
        pass
