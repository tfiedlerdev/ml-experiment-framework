from typing import Literal, Any, Optional
import torch
from torch.optim.optimizer import Optimizer
from src.datasets.joined_retina_dataset import (
    JoinedRetinaDataset,
    JoinedRetinaDatasetArgs,
)
from src.models.auto_sam_model import AutoSamModel, AutoSamModelArgs
from src.experiments.base_experiment import BaseExperiment, BaseExperimentArgs
from src.models.base_model import BaseModel
from src.args.yaml_config import YamlConfigModel
from src.datasets.base_dataset import BaseDataset
from src.optimizers.adam import AdamArgs
from src.schedulers.step_lr import StepLRArgs, create_steplr_scheduler
from typing import cast
import os
from pydantic import Field


class MultiDSVesselExperimentArgs(
    BaseExperimentArgs, AdamArgs, StepLRArgs, AutoSamModelArgs, JoinedRetinaDatasetArgs
):
    prompt_encoder_checkpoint: Optional[str] = Field(
        default=None, description="Path to prompt encoder checkpoint"
    )
    visualize_n_segmentations: int = Field(
        default=3, description="Number of images of test set to segment and visualize"
    )
    image_encoder_lr: Optional[float] = Field(
        default=None,
        description="Learning rate for image encoder, if None image encoder is frozen",
    )
    mask_decoder_lr: Optional[float] = Field(
        default=None,
        description="Learning rate for mask decoder,  if None image encoder is frozen",
    )
    prompt_encoder_lr: Optional[float] = Field(
        default=None,
        description="Learning rate for prompt encoder, if None, general learning rate is used",
    )


class MultiDsVesselExperiment(BaseExperiment):
    def __init__(self, config: dict[str, Any], yaml_config: YamlConfigModel):
        self.config = MultiDSVesselExperimentArgs(**config)

        self.ds = JoinedRetinaDataset.from_config(self.config, yaml_config)
        super().__init__(config, yaml_config)

    def get_name(self) -> str:
        return "multi_ds_vessel_experiment"

    def _create_dataset(
        self, split: Literal["train", "val", "test"] = "train"
    ) -> BaseDataset:
        return self.ds.get_split(split)

    def _create_model(self) -> BaseModel:
        image_encoder_no_grad = self.config.image_encoder_lr is None
        model = AutoSamModel(self.config, image_encoder_no_grad)
        if self.config.prompt_encoder_checkpoint is not None:
            print(
                f"loading prompt-encoder model from checkpoint {self.config.prompt_encoder_checkpoint}"
            )
            model.prompt_encoder.load_state_dict(
                torch.load(self.config.prompt_encoder_checkpoint, map_location="cuda"),
                strict=True,
            )
        return model

    @classmethod
    def get_args_model(cls):
        return MultiDSVesselExperimentArgs

    def create_optimizer(self) -> Optimizer:
        prompt_enc_params: dict = {
            "params": cast(AutoSamModel, self.model).prompt_encoder.parameters(),
        }
        if self.config.prompt_encoder_lr is not None:
            prompt_enc_params["lr"] = self.config.prompt_encoder_lr

        params = [prompt_enc_params]

        if self.config.image_encoder_lr is not None:
            params.append(
                {
                    "params": cast(
                        AutoSamModel, self.model
                    ).sam.image_encoder.parameters(),
                    "lr": self.config.image_encoder_lr,
                }
            )

        # Always add mask decoder to optimizer to allow for Automatic Mixed Precision to work even when mask decoder isn't trained
        # See bottom of https://chatgpt.com/share/675ae2c8-fff4-800c-8a5b-cecc352df76a
        params.append(
            {
                "params": cast(AutoSamModel, self.model).sam.mask_decoder.parameters(),
                "lr": (
                    self.config.mask_decoder_lr
                    if self.config.mask_decoder_lr is not None
                    else 0
                ),
            }
        )

        return torch.optim.Adam(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            eps=self.config.eps,
        )

    def create_scheduler(
        self, optimizer: Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        return create_steplr_scheduler(optimizer, self.config)

    def get_loss_name(self) -> str:
        return "dice+bce"

    def store_trained_model(self, trained_model: torch.nn.Module):
        model = cast(AutoSamModel, trained_model)
        torch.save(
            model.prompt_encoder.state_dict(),
            os.path.join(self.results_dir, "prompt_encoder.pt"),
        )
        torch.save(
            model.state_dict(),
            os.path.join(self.results_dir, "model.pt"),
        )

    def run_after_training(self, trained_model: BaseModel):
        model = cast(AutoSamModel, trained_model)

        def predict_visualize(split: Literal["train", "test"]):
            out_dir = os.path.join(self.results_dir, f"{split}_visualizations")
            os.makedirs(out_dir, exist_ok=True)
            ds = self.ds.get_split(split)
            print(
                f"\nCreating {self.config.visualize_n_segmentations} {split} segmentations"
            )
            for i in range(min(len(ds), self.config.visualize_n_segmentations)):
                sample = ds.get_file_refs()[i]
                out_path = os.path.join(out_dir, f"{i}.png")
                model.segment_and_write_image_from_file(
                    sample.img_path, out_path, gts_path=sample.gt_path
                )
                print(
                    f"{i+1}/{self.config.visualize_n_segmentations} {split} segmentations created\r",
                    end="",
                )

        predict_visualize("train")
        predict_visualize("test")
