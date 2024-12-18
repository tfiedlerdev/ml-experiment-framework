from dataclasses import dataclass
from pathlib import Path
import re
from src.models.auto_sam_model import SAMBatch


import cv2
from src.datasets.base_dataset import BaseDataset, Sample
from src.models.segment_anything.utils.transforms import ResizeLongestSide
from pydantic import BaseModel
from src.args.yaml_config import YamlConfigModel
from typing import Literal, Optional
import torch
from typing_extensions import Self
import os
from PIL import Image

from src.util.polyp_transform import get_polyp_transform


@dataclass
class RefugeSample(Sample):
    original_size: torch.Tensor
    image_size: torch.Tensor


@dataclass
class RefugeFileReference:
    img_path: str
    gt_path: str
    split: str


class RefugeDatasetArgs(BaseModel):
    """Define arguments for the dataset here, i.e. preprocessing related stuff etc"""

    target: Literal["cup", "disc"]


class RefugeDataset(BaseDataset):
    def __init__(
        self,
        config: RefugeDatasetArgs,
        yaml_config: YamlConfigModel,
        samples: Optional[list[RefugeFileReference]] = None,
    ):
        self.yaml_config = yaml_config
        self.config = config
        self.samples = self.load_data() if samples is None else samples
        pixel_mean, pixel_std = (
            self.yaml_config.fundus_pixel_mean,
            self.yaml_config.fundus_pixel_std,
        )
        self.sam_trans = ResizeLongestSide(
            self.yaml_config.fundus_resize_img_size,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )

    def __getitem__(self, index: int) -> RefugeSample:
        sample = self.samples[index]
        train_transform, test_transform = get_polyp_transform()

        augmentations = test_transform if sample.split == "test" else train_transform

        image = self.cv2_loader(sample.img_path, is_mask=False)
        gt = self.cv2_loader(sample.gt_path, is_mask=True)

        img, mask = augmentations(image, gt)

        original_size = tuple(img.shape[1:3])
        img, mask = self.sam_trans.apply_image_torch(
            torch.Tensor(img)
        ), self.sam_trans.apply_image_torch(torch.Tensor(mask))
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        image_size = tuple(img.shape[1:3])

        return RefugeSample(
            input=self.sam_trans.preprocess(img),
            target=self.sam_trans.preprocess(mask),
            original_size=torch.Tensor(original_size),
            image_size=torch.Tensor(image_size),
        )

    def __len__(self):
        return len(self.samples)

    def get_collate_fn(self):  # type: ignore
        def collate(samples: list[RefugeSample]):
            inputs = torch.stack([s.input for s in samples])
            targets = torch.stack([s.target for s in samples])
            original_size = torch.stack([s.original_size for s in samples])
            image_size = torch.stack([s.image_size for s in samples])
            return SAMBatch(
                inputs, targets, original_size=original_size, image_size=image_size
            )

        return collate

    def get_split(self, split: Literal["train", "val", "test"]) -> Self:

        return self.__class__(
            self.config,
            self.yaml_config,
            [sample for sample in self.samples if sample.split == split],
        )

    def load_data(self):

        train = self.load_data_for_split("train")
        val = self.load_data_for_split("val")
        test = self.load_data_for_split("test")

        return train + val + test

    def filter_files(self, old_images, old_gts):
        assert len(old_images) == len(old_gts)
        images = []
        gts = []
        for img_path, gt_path in zip(old_images, old_gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)

        return images, gts

    def load_data_for_split(self, split):
        dir_names = {
            "train": "Training-400",
            "val": "Validation-400",
            "test": "Test-400",
        }
        dir = os.path.join(self.yaml_config.refuge_dset_path, dir_names[split])

        images_and_masks_paths = [
            (
                str(
                    Path(self.yaml_config.refuge_dset_path)
                    / dir
                    / subdir
                    / f"{subdir}.jpg"
                ),
                str(
                    Path(self.yaml_config.refuge_dset_path)
                    / dir
                    / subdir
                    / f"{subdir}_seg_{self.config.target}_1.png"
                ),
            )
            for subdir in os.listdir(dir)
            if re.search("\d\d\d\d", subdir) is not None  # type: ignore
        ]

        return [
            RefugeFileReference(
                img_path=img,
                gt_path=mask,
                split=split,
            )
            for img, mask in images_and_masks_paths
        ]

    def cv2_loader(self, path, is_mask):
        if is_mask:
            img = cv2.imread(path, 0)
            img[img > 0] = 1
        else:
            img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return img
