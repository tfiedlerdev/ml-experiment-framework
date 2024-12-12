from dataclasses import dataclass
from typing import Literal, Optional

import torch
from src.args.yaml_config import YamlConfig
from src.util.polyp_transform import get_polyp_transform
from src.models.segment_anything.build_sam import (
    build_sam_vit_b,
    build_sam_vit_h,
    build_sam_vit_l,
)
from src.models.base_model import BaseModel, ModelOutput, Loss
from src.datasets.base_dataset import Batch
from pydantic import BaseModel as PDBaseModel
from torch.nn import BCELoss
from src.models.auto_sam_prompt_encoder.model_single import ModelEmb
from torch.nn import functional as F
import numpy as np


def get_dice_ji(predict, target):
    predict = predict + 1
    target = target + 1
    tp = np.sum(((predict == 2) * (target == 2)) * (target > 0))
    fp = np.sum(((predict == 2) * (target == 1)) * (target > 0))
    fn = np.sum(((predict == 1) * (target == 2)) * (target > 0))
    ji = float(np.nan_to_num(tp / (tp + fp + fn)))
    dice = float(np.nan_to_num(2 * tp / (2 * tp + fp + fn)))
    return dice, ji


class AutoSamModelArgs(PDBaseModel):
    sam_model: Literal["vit_h", "vit_l", "vit_b"]
    sam_checkpoint: str = "/dhc/groups/mp2024cl2/sam_vit_b.pth"
    hard_net_cp: str = "/dhc/groups/mp2024cl2/hardnet68.pth"
    hard_net_arch: int = 68
    depth_wise: bool = False
    Idim: int = 512


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


@dataclass
class SAMBatch(Batch):
    original_size: torch.Tensor
    image_size: torch.Tensor


@dataclass
class SAMSampleFileReference:
    img_path: str
    gt_path: str


# Source of most of this code: https://github.com/talshaharabany/AutoSAM
class AutoSamModel(BaseModel[SAMBatch]):
    def __init__(self, config: AutoSamModelArgs, image_encoder_no_grad=True):
        super().__init__()
        self.sam = sam_model_registry[config.sam_model](
            checkpoint=config.sam_checkpoint
        )
        self.bce_loss = BCELoss()
        self.prompt_encoder = ModelEmb(
            hard_net_arch=config.hard_net_arch,
            depth_wise=config.depth_wise,
            hard_net_cp=config.hard_net_cp,
        )
        self.config = config
        self.image_encoder_no_grad = image_encoder_no_grad

    def forward(self, batch: SAMBatch) -> ModelOutput:
        Idim = self.config.Idim
        orig_imgs_small = F.interpolate(
            batch.input, (Idim, Idim), mode="bilinear", align_corners=True
        )
        dense_embeddings = self.prompt_encoder(orig_imgs_small)
        masks = norm_batch(
            sam_call(
                batch.input, self.sam, dense_embeddings, self.image_encoder_no_grad
            )
        )

        return ModelOutput(masks)

    def compute_loss(self, outputs: ModelOutput, batch: SAMBatch) -> Loss:
        assert batch.target is not None
        size = outputs.logits.shape[2:]
        gts_sized = F.interpolate(batch.target.unsqueeze(dim=1), size, mode="nearest")

        bce = self.bce_loss.forward(outputs.logits, gts_sized)
        dice_loss = compute_dice_loss(outputs.logits, gts_sized)
        loss_value = bce + dice_loss

        input_size = tuple(batch.image_size[0][-2:].int().tolist())
        original_size = tuple(batch.original_size[0][-2:].int().tolist())
        gts = batch.target.unsqueeze(dim=0)
        masks = self.sam.postprocess_masks(
            outputs.logits, input_size=input_size, original_size=original_size
        )
        gts = self.sam.postprocess_masks(
            batch.target.unsqueeze(dim=0),
            input_size=input_size,
            original_size=original_size,
        )
        masks = F.interpolate(
            masks,
            (self.config.Idim, self.config.Idim),
            mode="bilinear",
            align_corners=True,
        )
        gts = F.interpolate(
            gts, (self.config.Idim, self.config.Idim), mode="bilinear"
        )  # was mode=nearest in original code
        masks[masks > 0.5] = 1
        masks[masks <= 0.5] = 0
        dice_score, IoU = get_dice_ji(
            masks.squeeze().detach().cpu().numpy(), gts.squeeze().detach().cpu().numpy()
        )

        return Loss(
            loss_value,
            {
                "dice+bce_loss": loss_value.detach().item(),
                "dice_loss": dice_loss.detach().item(),
                "bce_loss": bce.detach().item(),
                "dice_score": dice_score,
                "IoU": IoU,
            },
        )

    def segment_image(
        self,
        image: np.ndarray,
        pixel_mean: tuple[float, float, float],
        pixel_std: tuple[float, float, float],
    ):
        import cv2
        from .segment_anything.utils.transforms import ResizeLongestSide

        _, test_transform = get_polyp_transform()
        img, _ = test_transform(image, np.zeros_like(image))
        original_size = tuple(img.shape[1:3])

        transform = ResizeLongestSide(1024, pixel_mean, pixel_std)
        Idim = self.config.Idim

        image_tensor = transform.apply_image_torch(img)
        input_size = tuple(image_tensor.shape[1:3])
        input_images = transform.preprocess(image_tensor).unsqueeze(dim=0).cuda()

        orig_imgs_small = F.interpolate(
            input_images, (Idim, Idim), mode="bilinear", align_corners=True
        )
        dense_embeddings = self.prompt_encoder.forward(orig_imgs_small)
        with torch.no_grad():
            mask = norm_batch(sam_call(input_images, self.sam, dense_embeddings))

        mask = self.sam.postprocess_masks(
            mask, input_size=input_size, original_size=original_size
        )
        mask = mask.squeeze().cpu().numpy()
        mask = (255 * mask).astype(np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return image, mask

    def segment_image_from_file(self, image_path: str):
        import cv2

        image = cv2.cvtColor(
            cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
        )
        yaml_config = YamlConfig().config
        pixel_mean, pixel_std = (
            yaml_config.fundus_pixel_mean,
            yaml_config.fundus_pixel_std,
        )
        return self.segment_image(image, pixel_mean, pixel_std)

    def segment_and_write_image_from_file(
        self,
        image_path: str,
        output_path: str,
        mask_opacity: float = 0.4,
        gts_path: Optional[str] = None,
    ):
        import cv2
        from PIL import Image

        image, mask = self.segment_image_from_file(image_path)
        if gts_path is not None:
            with Image.open(gts_path) as im:
                gts = np.array(im.convert("RGB"))
        else:
            gts = np.zeros_like(mask)
        mask[mask > 255 / 2] = 255
        mask[mask <= 255 / 2] = 0
        overlay = (
            np.array(mask) * np.array([1, 0, 1]) + np.array(gts) * np.array([0, 1, 0])
        ).astype(image.dtype)
        output_image = cv2.addWeighted(
            image, 1 - mask_opacity, overlay, mask_opacity, 0
        )

        cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))


def compute_dice_loss(y_true, y_pred, smooth=1):
    alpha = 0.5
    beta = 0.5
    tp = torch.sum(y_true * y_pred, dim=(1, 2, 3))
    fn = torch.sum(y_true * (1 - y_pred), dim=(1, 2, 3))
    fp = torch.sum((1 - y_true) * y_pred, dim=(1, 2, 3))
    tversky_class = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return 1 - torch.mean(tversky_class)


def get_input_dict(imgs, original_sz, img_sz):
    batched_input = []
    for i, img in enumerate(imgs):
        input_size = tuple([int(x) for x in img_sz[i].squeeze().tolist()])
        original_size = tuple([int(x) for x in original_sz[i].squeeze().tolist()])
        singel_input = {
            "image": img,
            "original_size": original_size,
            "image_size": input_size,
            "point_coords": None,
            "point_labels": None,
        }
        batched_input.append(singel_input)
    return batched_input


def norm_batch(x):
    bs = x.shape[0]
    Isize = x.shape[-1]
    min_value = (
        x.view(bs, -1)
        .min(dim=1)[0]
        .repeat(1, 1, 1, 1)
        .permute(3, 2, 1, 0)
        .repeat(1, 1, Isize, Isize)
    )
    max_value = (
        x.view(bs, -1)
        .max(dim=1)[0]
        .repeat(1, 1, 1, 1)
        .permute(3, 2, 1, 0)
        .repeat(1, 1, Isize, Isize)
    )
    x = (x - min_value) / (max_value - min_value + 1e-6)
    return x


def sam_call(batched_input, sam, dense_embeddings, image_encoder_no_grad=True):
    with torch.set_grad_enabled(not image_encoder_no_grad):
        input_images = torch.stack([sam.preprocess(x) for x in batched_input], dim=0)
        image_embeddings = sam.image_encoder(input_images)
        sparse_embeddings_none, dense_embeddings_none = sam.prompt_encoder(
            points=None, boxes=None, masks=None
        )
    low_res_masks, iou_predictions = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings_none,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    return low_res_masks
