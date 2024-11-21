from typing import Literal, Any, Optional
from click import prompt
import torch
from torch.optim.optimizer import Optimizer
from src.datasets.ukbiobank_dataset import UkBiobankDataset, UkBiobankDatasetArgs
from src.models.auto_sam_model import AutoSamModel, AutoSamModelArgs
from src.experiments.base_experiment import BaseExperiment, BaseExperimentArgs
from src.models.base_model import BaseModel
from src.args.yaml_config import YamlConfigModel
from src.datasets.base_dataset import BaseDataset
from src.optimizers.adam import create_adam_optimizer, AdamArgs
from src.schedulers.step_lr import StepLRArgs, create_steplr_scheduler
from typing import cast
import os
from pydantic import Field


class UkBiobankExperimentArgs(
    BaseExperimentArgs, AdamArgs, StepLRArgs, AutoSamModelArgs, UkBiobankDatasetArgs
):
    prompt_encoder_checkpoint: Optional[str] = Field(
        default=None, description="Path to prompt encoder checkpoint"
    )
    visualize_n_segmentations: int = Field(
        default=3, description="Number of images of test set to segment and visualize"
    )
    image_encoder_lr: Optional[float] = Field(
        default=None, description="Learning rate for image encoder"
    )
    mask_decoder_lr: Optional[float] = Field(
        default=None, description="Learning rate for mask decoder"
    )
    prompt_encoder_lr: Optional[float] = Field(
        default=None, description="Learning rate for prompt encoder"
    )


class UkBioBankExperiment(BaseExperiment):
    def __init__(self, config: dict[str, Any], yaml_config: YamlConfigModel):
        self.config = UkBiobankExperimentArgs(**config)

        self.ds = UkBiobankDataset(
            config=self.config, yaml_config=yaml_config, with_masks=True
        )
        super().__init__(config, yaml_config)

    def get_name(self) -> str:
        return "uk_biobank_experiment"

    def _create_dataset(
        self, split: Literal["train", "val", "test"] = "train"
    ) -> BaseDataset:
        return self.ds.get_split(split)

    def _create_model(self) -> BaseModel:
        model = AutoSamModel(self.config, grad_only_prompt_encoder=False)
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
        return UkBiobankExperimentArgs

    def create_optimizer(self) -> Optimizer:
        def get_trainable_params():
            return [
                {
                    "params": cast(
                        AutoSamModel, self.model
                    ).sam.image_encoder.parameters(),
                    "lr": (
                        self.config.image_encoder_lr
                        if self.config.image_encoder_lr is not None
                        else self.config.learning_rate
                    ),
                },
                {
                    "params": cast(
                        AutoSamModel, self.model
                    ).sam.mask_decoder.parameters(),
                    "lr": (
                        self.config.mask_decoder_lr
                        if self.config.mask_decoder_lr is not None
                        else self.config.learning_rate
                    ),
                },
                {
                    "params": cast(
                        AutoSamModel, self.model
                    ).sam.prompt_encoder.parameters(),
                    "lr": (
                        self.config.prompt_encoder_lr
                        if self.config.prompt_encoder_lr is not None
                        else self.config.learning_rate
                    ),
                },
            ]

        return torch.optim.Adam(
            get_trainable_params(),
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
                sample = ds.samples[i]
                out_path = os.path.join(out_dir, f"{i}.png")
                model.segment_and_write_image_from_file(
                    str(sample.img_path),
                    out_path,
                    gts_path=str(sample.gt_path),
                )
                print(
                    f"{i+1}/{self.config.visualize_n_segmentations} {split} segmentations created\r",
                    end="",
                )

        predict_visualize("train")
        predict_visualize("test")