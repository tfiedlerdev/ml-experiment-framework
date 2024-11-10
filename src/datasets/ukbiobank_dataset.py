from dataclasses import dataclass
from math import floor
import os
from pathlib import Path
from typing import Literal, Optional
import numpy as np
import torch
import cv2
from typing_extensions import Self
from src.args.yaml_config import YamlConfigModel
from src.datasets.base_dataset import BaseDataset, Batch, Sample
from pydantic import BaseModel

from src.datasets.refuge_dataset import get_polyp_transform
from src.models.segment_anything.utils.transforms import ResizeLongestSide


class UkBiobankDatasetArgs(BaseModel):
    train_percentage: float = 0.8
    val_percentage: float = 0.15
    test_percentage: float = 0.05


@dataclass
class BiobankSampleReference:
    img_path: Path
    gt_path: Path | None
    split: str


@dataclass
class BiobankSample(Sample):
    split: str
    original_size: torch.Tensor
    image_size: torch.Tensor


@dataclass
class BiobankBatch(Batch):
    original_size: torch.Tensor
    image_size: torch.Tensor


class UkBiobankDataset(BaseDataset):

    def __init__(
        self,
        config: UkBiobankDatasetArgs,
        yaml_config: YamlConfigModel,
        samples: Optional[list[BiobankSampleReference]] = None,
        image_enc_img_size=1024,
        with_masks=False,
    ):
        self.config = config
        self.yaml_config = yaml_config
        self.with_masks = with_masks
        self.samples = self.load_data() if samples is None else samples
        self.sam_trans = ResizeLongestSide(image_enc_img_size)

    def __getitem__(self, index: int) -> BiobankSample:
        sample = self.samples[index]
        train_transform, test_transform = get_polyp_transform()

        augmentations = test_transform if sample.split == "test" else train_transform
        image = self.cv2_loader(sample.img_path, is_mask=False)
        gt = (
            self.cv2_loader(sample.gt_path, is_mask=True)
            if self.with_masks
            else np.zeros_like(image)
        )

        img, mask = augmentations(image, gt)

        mask = self.sam_trans.apply_image_torch(torch.Tensor(mask))
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        original_size = tuple(img.shape[1:3])
        img = self.sam_trans.apply_image_torch(torch.Tensor(img))
        image_size = tuple(img.shape[1:3])

        return BiobankSample(
            input=self.sam_trans.preprocess(img),
            target=mask,
            original_size=torch.Tensor(original_size),
            image_size=torch.Tensor(image_size),
            split=sample.split,
        )

    def __len__(self):
        return len(self.samples)

    def get_collate_fn(self):  # type: ignore
        def collate(samples: list[BiobankSample]):
            inputs = torch.stack([s.input for s in samples])
            targets = torch.stack([s.target for s in samples])
            original_size = torch.stack([s.original_size for s in samples])
            image_size = torch.stack([s.image_size for s in samples])
            return BiobankBatch(
                inputs, targets, original_size=original_size, image_size=image_size
            )

        return collate

    def get_split(self, split: Literal["train", "val", "test"]) -> Self:
        return self.__class__(
            self.config,
            self.yaml_config,
            [sample for sample in self.samples if sample.split == split],
            with_masks=self.with_masks,
        )

    def load_data(self) -> list[BiobankSampleReference]:
        sample_folder = Path(self.yaml_config.ukbiobank_data_dir)
        mask_folder = Path(self.yaml_config.ukbiobank_masks_dir)
        sample_paths = [
            (sample_folder / path, mask_folder / path if self.with_masks else None)
            for path in os.listdir(sample_folder)
            if not Path(path).is_file() and path.endswith(".png")
        ]

        train = self.load_data_for_split("train", sample_paths)
        val = self.load_data_for_split("val", sample_paths)
        test = self.load_data_for_split("test", sample_paths)

        return train + val + test

    def load_data_for_split(
        self, split, sample_paths: list[tuple[Path, Path | None]]
    ) -> list[BiobankSampleReference]:
        index_offset = (
            0
            if split == "train"
            else (
                floor(len(sample_paths) * self.config.train_percentage)
                if split == "val"
                else floor(
                    len(sample_paths)
                    * (self.config.train_percentage + self.config.val_percentage)
                )
            )
        )
        length = (
            floor(len(sample_paths) * self.config.train_percentage)
            if split == "train"
            else (
                floor(len(sample_paths) * self.config.val_percentage)
                if split == "val"
                else floor(len(sample_paths) * self.config.test_percentage)
            )
        )

        return [
            BiobankSampleReference(img_path=img_path, gt_path=gt_path, split=split)
            for img_path, gt_path in sample_paths[index_offset : index_offset + length]
        ]

    def cv2_loader(self, path, is_mask):
        if is_mask:
            img = cv2.imread(path, 0)
            img[img > 0] = 1
        else:
            img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return img
