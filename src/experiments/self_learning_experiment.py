from typing import Literal, Any, Optional
from click import prompt
import torch
from torch.optim.optimizer import Optimizer
from src.datasets.joined_retina_dataset import (
    JoinedRetinaDataset,
    JoinedRetinaDatasetArgs,
)
from torch.utils.data import DataLoader
from src.datasets.ukbiobank_dataset import UkBiobankDataset, UkBiobankDatasetArgs
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


class SelfLearningExperimentArgs(
    BaseExperimentArgs,
    AdamArgs,
    StepLRArgs,
    AutoSamModelArgs,
    UkBiobankDatasetArgs,
    JoinedRetinaDatasetArgs,
):
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
    teacher_student_update_ratio: float = Field(
        default=0.2, description="Ratio of teacher to student update"
    )
    ema_decay_origin: float = Field(
        default=0.999, description="Exponential moving average decay"
    )
    secondary_batch_size: int = Field(
        default=16, description="Batch size for unlabeled data"
    )


class SelfLearningExperiment(BaseExperiment):
    def __init__(self, config: dict[str, Any], yaml_config: YamlConfigModel):
        self.config = SelfLearningExperimentArgs(**config)

        # Setting up labeled dataset and unlabeled dataset
        self.ds_w_labels = JoinedRetinaDataset.from_config(self.config, yaml_config)
        self.ds_wo_labels = UkBiobankDataset(
            config=self.config,
            yaml_config=yaml_config,
            with_masks=False,
            random_augmentation_for_all_splits=True,
        )
        super().__init__(config, yaml_config)

        self.unlabeled_loader = DataLoader(
            self.ds_wo_labels,
            batch_size=self.config.secondary_batch_size,
            shuffle=True,
            collate_fn=self.ds_wo_labels.get_collate_fn(),
        )

        # Setting up student model
        # Same architecture, same initial checkpoint
        self.student_model = self._create_model()
        if self.base_config.from_checkpoint is not None:
            self.student_model.load_state_dict(
                torch.load(self.base_config.from_checkpoint, map_location="cuda"),
                strict=True,
            )
        self.student_model.to("cuda")

    def get_name(self) -> str:
        return "self_learning_experiment"

    def _create_trainer(self):
        from src.train.self_trainer import SelfTrainer

        return SelfTrainer(self)

    def _create_dataset(
        self, split: Literal["train", "val", "test"] = "train"
    ) -> BaseDataset:
        return self.ds_w_labels.get_split(split)

    def _create_model(self) -> BaseModel:
        model = AutoSamModel(self.config, image_encoder_no_grad=False)
        return model

    @classmethod
    def get_args_model(cls):
        return SelfLearningExperimentArgs

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
                    ).prompt_encoder.parameters(),
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

    def run(self):
        # Execute finetuning
        super().run()

    def run_after_training(self, trained_model: BaseModel):
        model = cast(AutoSamModel, trained_model)

        def predict_visualize(split: Literal["train", "test"]):
            out_dir = os.path.join(self.results_dir, f"{split}_visualizations")
            os.makedirs(out_dir, exist_ok=True)
            ds = self.ds_wo_labels.get_split(split)
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
