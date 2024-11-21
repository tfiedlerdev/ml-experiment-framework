from dataclasses import dataclass
from pathlib import Path
import re
from src.models.auto_sam_model import SAMBatch, SAMSampleFileReference
from src.datasets.refuge_dataset import get_polyp_transform
import src.util.transforms_shir as transforms
import numpy as np
import cv2
from src.datasets.base_dataset import BaseDataset, Batch, Sample
from src.models.segment_anything.utils.transforms import ResizeLongestSide
from torchvision.datasets import MNIST
from pydantic import BaseModel, Field
from src.args.yaml_config import YamlConfigModel
from typing import Callable, Literal, Optional
from math import floor
import torch
from typing_extensions import Self
import os
from PIL import Image

from src.util.image_util import calculate_rgb_mean_std


@dataclass
class ChaseDb1Sample(Sample):
    original_size: torch.Tensor
    image_size: torch.Tensor


@dataclass
class ChaseDb1FileReference(SAMSampleFileReference):
    id: str
    split: str


class ChaseDb1DatasetArgs(BaseModel):
    """Define arguments for the dataset here, i.e. preprocessing related stuff etc"""

    chasedb1_train_percentage: float = Field(
        default=0.8,
        description="Percentage of data to use for training. Other data will be assigned to val and, if enabled, test.",
    )


class ChaseDb1Dataset(BaseDataset):
    def __init__(
        self,
        config: ChaseDb1DatasetArgs,
        yaml_config: YamlConfigModel,
        samples: Optional[list[ChaseDb1FileReference]] = None,
        image_enc_img_size=1024,
    ):
        self.yaml_config = yaml_config
        self.config = config
        self.samples = self.load_data() if samples is None else samples
        pixel_mean, pixel_std = calculate_rgb_mean_std(
            [s.img_path for s in self.samples],
            os.path.join(yaml_config.cache_dir, "chasedb1_mean_std.pkl"),
        )
        self.sam_trans = ResizeLongestSide(
            image_enc_img_size, pixel_mean=pixel_mean, pixel_std=pixel_std
        )

    def __getitem__(self, index: int) -> ChaseDb1Sample:
        sample = self.samples[index]
        train_transform, test_transform = get_polyp_transform()

        augmentations = train_transform if sample.split == "train" else test_transform

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

        return ChaseDb1Sample(
            input=self.sam_trans.preprocess(img),
            target=self.sam_trans.preprocess(mask),
            original_size=torch.Tensor(original_size),
            image_size=torch.Tensor(image_size),
        )

    def __len__(self):
        return len(self.samples)

    def get_collate_fn(self):  # type: ignore
        def collate(samples: list[ChaseDb1Sample]):
            inputs = torch.stack([s.input for s in samples])
            targets = torch.stack([s.target for s in samples])
            original_size = torch.stack([s.original_size for s in samples])
            image_size = torch.stack([s.image_size for s in samples])
            return SAMBatch(
                inputs, targets, original_size=original_size, image_size=image_size
            )

        return collate

    def get_split(self, split: Literal["train", "val", "test"]) -> Self:
        if split == "test":
            # we only have train and val split here
            split = "val"
        samples = [s for s in self.samples if s.split == split]
        return self.__class__(
            self.config,
            self.yaml_config,
            samples,
        )

    def load_data(self):
        dir = self.yaml_config.chasedb1_dset_path

        imgs = [f for f in os.listdir(dir) if f.endswith(".jpg")]

        refs = []
        train_ratio = self.config.chasedb1_train_percentage
        for i, img_file_name in enumerate(imgs):
            is_train = i / len(imgs) < train_ratio
            is_val = (
                i / len(imgs) < train_ratio + (1 - train_ratio) / 2 and not is_train
            )
            split = "train" if is_train else "val" if is_val else "test"
            img = str(Path(dir) / img_file_name)
            gt1 = str(Path(dir) / img_file_name.replace(".jpg", "_1stHO.png"))
            gt2 = str(Path(dir) / img_file_name.replace(".jpg", "_2ndHO.png"))
            refs.append(
                ChaseDb1FileReference(
                    img_path=img,
                    gt_path=gt1,
                    id=img_file_name,
                    split=split,
                )
            )
            refs.append(
                ChaseDb1FileReference(
                    img_path=img,
                    gt_path=gt2,
                    id=img_file_name,
                    split=split,
                )
            )

        return refs

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

    def cv2_loader(self, path, is_mask):
        if is_mask:
            with Image.open(path) as im:
                img = np.array(im.convert("L"))
            img[img > 0] = 1
        else:
            img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return img
