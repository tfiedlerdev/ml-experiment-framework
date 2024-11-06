from dataclasses import dataclass
from math import floor
import os
from pathlib import Path
from typing import Literal, Optional
import torch
from typing_extensions import Self
from args.yaml_config import YamlConfigModel
from datasets.base_dataset import BaseDataset, Sample
from pydantic import BaseModel
from PIL import Image
from torchvision import transforms


class UkBiobankDatasetArgs(BaseModel):
    train_percentage: float = 0.8
    val_percentage: float = 0.15
    test_percentage: float = 0.05


@dataclass
class BiobankSampleReference:
    img_path: Path


class UkBiobankDataset(BaseDataset):

    def __init__(
        self,
        config: UkBiobankDatasetArgs,
        yaml_config: YamlConfigModel,
        samples: Optional[list[BiobankSampleReference]] = None,
    ):
        self.config = config
        self.yaml_config = yaml_config
        self.samples = self.load_data() if samples is None else samples

    def __getitem__(self, index: int) -> Sample:
        return self._load_sample(self.samples[index].img_path)

    def __len__(self):
        return len(self.samples)

    def get_split(self, split: Literal["train", "val", "test"]) -> Self:
        index_offset = (
            0
            if split == "train"
            else (
                floor(len(self.samples) * self.config.train_percentage)
                if split == "val"
                else floor(
                    len(self.samples)
                    * (self.config.train_percentage + self.config.val_percentage)
                )
            )
        )
        length = (
            floor(len(self.samples) * self.config.train_percentage)
            if split == "train"
            else (
                floor(len(self.samples) * self.config.val_percentage)
                if split == "val"
                else floor(len(self.samples) * self.config.test_percentage)
            )
        )
        return self.__class__(
            self.config,
            self.yaml_config,
            self.samples[index_offset : index_offset + length],
        )

    def load_data(self) -> list[BiobankSampleReference]:
        sample_folder = Path(self.yaml_config.ukbiobank_data_dir)
        samples = [
            BiobankSampleReference(img_path=sample_folder / path)
            for path in os.listdir(sample_folder)
            if not Path(path).is_file() and path.endswith(".png")
        ]
        return samples

    def _load_sample(self, sample_path: Path) -> Sample:
        input = Image.open(sample_path)
        input_tensor = transforms.ToTensor()(input)
        return Sample(input=input_tensor, target=None)
