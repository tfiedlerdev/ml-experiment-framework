from typing import Literal

import torch
from src.datasets.refuge_dataset import RefugeBatch
from src.models.segment_anything.build_sam import (
    build_sam_vit_b,
    build_sam_vit_h,
    build_sam_vit_l,
)
from src.models.base_model import BaseModel, ModelOutput, Loss
from src.datasets.base_dataset import Batch
from src.util.nn_helper import create_fully_connected, ACTIVATION_FUNCTION
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
    sam_checkpoint: str = "./sam_vit_b.pth"
    hard_net_cp: str = "./hardnet68.pth"
    hard_net_arch: int = 68
    depth_wise: bool = False
    Idim: int = 512


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


# Source of most of this code: https://github.com/talshaharabany/AutoSAM
class AutoSamModel(BaseModel[RefugeBatch]):
    def __init__(self, config: AutoSamModelArgs):
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

    def forward(self, batch: RefugeBatch) -> ModelOutput:
        Idim = self.config.Idim
        orig_imgs_small = F.interpolate(
            batch.input, (Idim, Idim), mode="bilinear", align_corners=True
        )
        dense_embeddings = self.prompt_encoder(orig_imgs_small)
        batched_input = get_input_dict(
            batch.input, batch.original_size, batch.image_size
        )
        masks = norm_batch(sam_call(batched_input, self.sam, dense_embeddings))

        return ModelOutput(masks)

    def compute_loss(self, outputs: ModelOutput, batch: RefugeBatch) -> Loss:
        assert batch.target is not None
        size = outputs.logits.shape[2:]
        gts_sized = F.interpolate(batch.target.unsqueeze(dim=1), size, mode="nearest")

        bce = self.bce_loss.forward(outputs.logits, gts_sized)
        dice_loss = compute_dice_loss(outputs.logits, gts_sized)
        loss_value = bce + dice_loss

        input_size = tuple([int(x) for x in batch.image_size[0].squeeze().tolist()])
        original_size = tuple(
            [int(x) for x in batch.original_size[0].squeeze().tolist()]
        )
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
        gts = F.interpolate(gts, (self.config.Idim, self.config.Idim), mode="nearest")
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

    def segment_image(self, image: np.ndarray):
        import cv2
        from .segment_anything.utils.transforms import ResizeLongestSide

        def sam_call(batched_input, sam, dense_embeddings):
            with torch.no_grad():
                input_images = torch.stack(
                    [sam.preprocess(x) for x in batched_input], dim=0
                )
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

        original_size = image.shape[:2]
        transform = ResizeLongestSide(1024)
        Idim = 512
        image_tensor = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        image_tensor = transform.apply_image_torch(image_tensor)

        input_images = transform.preprocess(image_tensor).unsqueeze(dim=0)
        input_images = image_tensor.unsqueeze(dim=0).cuda()

        orig_imgs_small = F.interpolate(
            input_images, (Idim, Idim), mode="bilinear", align_corners=True
        )
        dense_embeddings = self.prompt_encoder.forward(orig_imgs_small)

        mask = sam_call(input_images, self.sam, dense_embeddings)
        input_size = image_tensor.shape[1:]
        mask = self.sam.postprocess_masks(
            mask, input_size=input_size, original_size=original_size
        )
        mask = mask.squeeze().cpu().numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        mask = (255 * mask).astype(np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return image, mask

    def segment_image_from_file(self, image_path: str):
        import cv2

        image = cv2.imread(image_path)
        return self.segment_image(image)

    def segment_and_write_image_from_file(
        self, image_path: str, output_path: str, mask_opacity: float = 0.4
    ):
        import cv2

        image, mask = self.segment_image_from_file(image_path)
        overlay = (np.array(mask) * np.array([1, 1, 0])).astype(image.dtype)
        output_image = cv2.addWeighted(
            image, 1 - mask_opacity, overlay, mask_opacity, 0
        )
        cv2.imwrite(output_path, output_image)


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


def sam_call(batched_input, sam, dense_embeddings):
    with torch.no_grad():
        input_images = torch.stack(
            [sam.preprocess(x["image"]) for x in batched_input], dim=0
        )
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
