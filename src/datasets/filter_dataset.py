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


@dataclass
class FilterSample(Sample):
    file_path: str


@dataclass
class FilterBatch(Batch):
    file_paths: list[str]


class FilterDatasetArgs(BaseModel):
    """Define arguments for the dataset here, i.e. preprocessing related stuff etc"""

    train_percentage: float = 0.75
    val_percentage: float = 0.15
    test_percentage: float = 0.1


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

    def __getitem__(self, index: int) -> FilterSample:
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

        return FilterSample(
            input=torch.Tensor(preprocess_img),
            target=torch.Tensor([sample.label == 0, sample.label == 1]),
            file_path=sample.img_path,
        )

    def __len__(self):
        return len(self.samples)

    def get_collate_fn(self):  # type: ignore
        def collate(samples: list[FilterSample]):
            inputs = torch.stack([s.input for s in samples])
            targets = torch.stack([s.target for s in samples])
            file_paths = [s.file_path for s in samples]
            return FilterBatch(inputs, targets, file_paths)

        return collate

    def get_split(self, split: Literal["train", "val", "test"]) -> Self:
        train_start_idx = 0
        val_start_idx = floor(len(self.samples) * self.config.train_percentage)
        test_start_idx = floor(
            len(self.samples)
            * (self.config.train_percentage + self.config.val_percentage)
        )

        start_idx = (
            train_start_idx
            if split == "train"
            else val_start_idx if split == "val" else test_start_idx
        )
        end_idx = (
            val_start_idx
            if split == "train"
            else test_start_idx if split == "val" else len(self.samples)
        )

        return self.__class__(
            self.config,
            self.yaml_config,
            self.samples[start_idx:end_idx],
        )

    def load_data(self) -> list[FilterFileReference]:
        dir = self.yaml_config.filter_dset_path

        good_sample_filepath = os.path.join(dir, "good.txt")
        bad_sample_filepath = os.path.join(dir, "bad.txt")

        samples = []

        for i, path in enumerate([bad_sample_filepath, good_sample_filepath]):
            with open(path, "r") as f:
                sample_lines = f.readlines()

                for line in sample_lines:
                    line = line.strip()
                    samples.append(
                        FilterFileReference(
                            line,
                            i,
                        )
                    )
        seed(42)
        shuffle(samples)
        return samples
