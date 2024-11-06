from src.datasets.base_dataset import BaseDataset, Sample
from torchvision.datasets import MNIST
from pydantic import BaseModel
from src.args.yaml_config import YamlConfigModel
from typing import Literal, Optional
from math import floor
import torchvision.transforms as transforms
import torch
from typing_extensions import Self
import os


class MnistDatasetArgs(BaseModel):
    """Define arguments for the dataset here, i.e. preprocessing related stuff etc"""

    pass


class MnistDataset(BaseDataset):
    def __init__(
        self,
        config: MnistDatasetArgs,
        yaml_config: YamlConfigModel,
        samples: Optional[list[Sample]] = None,
    ):
        self.yaml_config = yaml_config
        self.config = config
        self.samples = self.load_data() if samples is None else samples

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

    def get_split(self, split: Literal["train", "val", "test"]) -> Self:
        index_offset = (
            0
            if split == "train"
            else (
                floor(len(self.samples) * 0.8)
                if split == "val"
                else floor(len(self.samples) * 0.95)
            )
        )
        length = (
            floor(len(self.samples) * 0.8)
            if split == "train"
            else (
                floor(len(self.samples) * 0.15)
                if split == "val"
                else floor(len(self.samples) * 0.05)
            )
        )
        return self.__class__(
            self.config,
            self.yaml_config,
            self.samples[index_offset : index_offset + length],
        )

    def load_data(self):
        mnist_data = MNIST(
            os.path.join(self.yaml_config.cache_dir, "mnist"),
            download=True,
        )
        image_transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Convert image to tensor
                transforms.Lambda(lambda x: x.view(-1)),  # Flatten the image to 1D
            ]
        )
        labels_matrix = torch.eye(10)
        target_transform = transforms.Lambda(lambda y: labels_matrix[y])
        return [
            Sample(image_transform(image), target_transform(target))
            for image, target in mnist_data
        ]
