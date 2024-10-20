from src.datasets.base_dataset import BaseDataset, Sample
from torchvision.datasets import MNIST
from pydantic import BaseModel
from src.args.yaml_config import YamlConfigModel
from typing import Literal
from math import floor
import torchvision.transforms as transforms
import torch


class MnistDatasetArgs(BaseModel):
    """Define arguments for the dataset here, i.e. preprocessing related stuff etc"""

    pass


class MnistDataset(BaseDataset):
    def __init__(
        self,
        all_data: MNIST,
        config: MnistDatasetArgs,
        yaml_config: YamlConfigModel,
        split: Literal["train", "val", "test"],
    ):
        self.index_offset = (
            0
            if split == "train"
            else floor(len(all_data) * 0.8)
            if split == "val"
            else floor(len(all_data) * 0.95)
        )
        self.length = (
            floor(len(all_data) * 0.8)
            if split == "train"
            else floor(len(all_data) * 0.15)
            if split == "val"
            else floor(len(all_data) * 0.05)
        )
        self.ds = all_data
        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Convert image to tensor
                transforms.Lambda(lambda x: x.view(-1)),  # Flatten the image to 1D
            ]
        )
        labels_matrix = torch.eye(10)
        self.target_transform = transforms.Lambda(lambda y: labels_matrix[y])

    def __getitem__(self, index: int) -> Sample:
        image, target = self.ds[index + self.index_offset]
        return Sample(self.image_transform(image), self.target_transform(target))

    def __len__(self):
        return self.length
