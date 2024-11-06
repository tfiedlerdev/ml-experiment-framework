from abc import abstractmethod
from typing import Any, Callable, NamedTuple, Optional, Literal
from typing_extensions import Self

import torch
from torch.utils.data import Dataset
from torch._prims_common import DeviceLikeType
from dataclasses import dataclass


@dataclass
class Sample:
    input: torch.Tensor
    target: Any


@dataclass
class Batch:
    input: torch.Tensor
    target: Optional[torch.Tensor]

    def to(self, device: DeviceLikeType):
        self.input = self.input.to(device)
        self.target = self.target.to(device) if self.target is not None else None

        if hasattr(self, "__dict__"):
            # Putting all tensors of potential subclass attributes to device
            for key, value in self.__dict__.items():
                if isinstance(value, torch.Tensor):
                    self.__setattr__(key, value.to(device))
        return self

    def cuda(self):
        return self.to("cuda")

    def copy_and_change(self, **diff):
        copy = self.__class__(**diff)
        for key, value in self.__dict__.items():
            copy.__setattr__(key, value)

        return copy


class BaseDataset(Dataset):
    @abstractmethod
    def __getitem__(self, index: int) -> Sample:
        raise NotImplementedError(
            "This method should be overridden in a subclass of BaseDataset"
        )

    def get_collate_fn(self) -> Callable[[list[Sample]], Batch]:
        def collate(samples: list[Sample]):
            inputs = torch.stack([s.input for s in samples])
            targets = torch.stack([s.target for s in samples])
            return Batch(inputs, targets)

        return collate

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def get_split(self, split: Literal["train", "val", "test"]) -> Self:
        pass
