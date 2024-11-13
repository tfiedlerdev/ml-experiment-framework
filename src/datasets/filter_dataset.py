from dataclasses import dataclass
from pathlib import Path
from random import random, seed, shuffle

from numpy import zeros_like
from src.datasets.base_dataset import BaseDataset, Batch, Sample
from src.models.segment_anything.utils.transforms import ResizeLongestSide
from torchvision.datasets import MNIST
from pydantic import BaseModel
from src.args.yaml_config import YamlConfigModel
from typing import Callable, Literal, Optional
from math import floor
import torch
from typing_extensions import Self
import os
from PIL import Image
from torchvision import transforms


@dataclass
class FilterFileReference:
    img_path: str
    label: int


class FilterDatasetArgs(BaseModel):
    """Define arguments for the dataset here, i.e. preprocessing related stuff etc"""

    train_percentage: float = 0.8
    val_percentage: float = 0.2


class FilterDataset(BaseDataset):
    def __init__(
        self,
        config: FilterDatasetArgs,
        yaml_config: YamlConfigModel,
        samples: Optional[list[FilterFileReference]] = None,
    ):
        self.yaml_config = yaml_config
        self.config = config
        self.samples = self.load_data() if samples is None else samples

    def __getitem__(self, index: int) -> Sample:
        sample = self.samples[index]

        image = Image.open(sample.img_path)

        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        preprocess_img = preprocess(image)

        return Sample(
            input=torch.Tensor(preprocess_img),
            target=torch.Tensor([sample.label == 0, sample.label == 1]),
        )

    def __len__(self):
        return len(self.samples)

    def get_collate_fn(self):  # type: ignore
        def collate(samples: list[Sample]):
            inputs = torch.stack([s.input for s in samples])
            targets = torch.stack([s.target for s in samples])
            return Batch(inputs, targets)

        return collate

    def get_split(self, split: Literal["train", "val", "test"]) -> Self:
        seed(42)
        shuffle(self.samples)
        index_offset = (
            0
            if split == "train"
            else floor(len(self.samples) * self.config.train_percentage)
        )
        length = (
            floor(len(self.samples) * self.config.train_percentage)
            if split == "train"
            else floor(len(self.samples) * self.config.val_percentage)
        )

        return self.__class__(
            self.config,
            self.yaml_config,
            self.samples[index_offset : index_offset + length],
        )

    def load_data(self) -> list[FilterFileReference]:
        dir = self.yaml_config.filter_dset_path

        good_sample_filepath = os.path.join(dir, "good.txt")
        bad_sample_filepath = os.path.join(dir, "bad.txt")

        file_samples = []

        for i, path in enumerate([bad_sample_filepath, good_sample_filepath]):
            with open(path, "r") as f:
                sample_lines = f.readlines()
                for line in sample_lines:
                    line = line.strip()
                    file_samples.append(FilterFileReference(line, i))

        return file_samples
