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
    test: bool


@dataclass
class FilterSample(Sample):
    file_path: str


@dataclass
class FilterBatch(Batch):
    file_paths: list[str]


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
        if split != "test":
            train_val_samples = [sample for sample in self.samples if not sample.test]
            samples = (
                train_val_samples[
                    : floor(len(train_val_samples) * self.config.train_percentage)
                ]
                if split == "train"
                else self.samples[
                    floor(len(train_val_samples) * self.config.train_percentage) :
                ]
            )
        else:
            samples = [sample for sample in self.samples if sample.test]
        return self.__class__(
            self.config,
            self.yaml_config,
            samples,
        )

    def load_data(self) -> list[FilterFileReference]:
        dir = self.yaml_config.filter_dset_path

        good_sample_filepath = os.path.join(dir, "good.txt")
        bad_sample_filepath = os.path.join(dir, "bad.txt")
        test_sample_filepath = os.path.join(dir, "good_biobank.txt")

        train_val_samples = []
        test_file_samples = []

        with open(good_sample_filepath, "r") as f:
            sample_lines = f.readlines()

            for line in sample_lines:
                line = line.strip()
                train_val_samples.append(FilterFileReference(line, 1, False))

        with open(bad_sample_filepath, "r") as f:
            sample_lines = f.readlines()

            for i, line in enumerate(sample_lines):
                line = line.strip()
                is_test = i > len(sample_lines) - 20
                if is_test:
                    test_file_samples.append(FilterFileReference(line, 0, True))
                else:
                    train_val_samples.append(FilterFileReference(line, 0, False))

        with open(test_sample_filepath, "r") as f:
            sample_lines = f.readlines()

            for line in sample_lines:
                line = line.strip()
                test_file_samples.append(FilterFileReference(line, 1, True))

        seed(42)
        shuffle(train_val_samples)
        return train_val_samples + test_file_samples
