from typing import Literal, Any, Optional
import torch
from torch.optim.optimizer import Optimizer
from src.datasets.refuge_dataset import RefugeDataset, RefugeDatasetArgs
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


class RefugeExperimentArgs(
    BaseExperimentArgs, AdamArgs, StepLRArgs, RefugeDatasetArgs, AutoSamModelArgs
):
    prompt_encoder_checkpoint: Optional[str] = Field(
        default=None, description="Path to prompt encoder checkpoint"
    )
    visualize_n_segmentations: int = Field(
        default=3, description="Number of images of test set to segment and visualize"
    )


class RefugeExperiment(BaseExperiment):
    def __init__(self, config: dict[str, Any], yaml_config: YamlConfigModel):
        self.config = RefugeExperimentArgs(**config)
        self.retina_data = RefugeDataset(config=self.config, yaml_config=yaml_config)
        super().__init__(config, yaml_config)

    def get_name(self) -> str:
        return "refuge_experiment"

    def _create_dataset(
        self, split: Literal["train", "val", "test"] = "train"
    ) -> BaseDataset:
        return self.retina_data.get_split(split)

    def _create_model(self) -> BaseModel:
        model = AutoSamModel(self.config)
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
        return RefugeExperimentArgs

    def create_optimizer(self) -> Optimizer:
        return create_adam_optimizer(self.model, self.config)

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

    def run_after_training(self, trained_model: BaseModel):
        model = cast(AutoSamModel, trained_model)
        out_dir = os.path.join(self.results_dir, "test_visualizations")
        os.makedirs(out_dir, exist_ok=True)
        ds = cast(RefugeDataset, self._create_dataset("test"))
        print(f"\nCreating {self.config.visualize_n_segmentations} test segmentations")
        for i in range(min(len(ds.samples), self.config.visualize_n_segmentations)):
            sample = ds.samples[i]
            out_path = os.path.join(out_dir, f"{i}.png")
            model.segment_and_write_image_from_file(sample.img_path, out_path)
            print(
                f"{i+1}/{self.config.visualize_n_segmentations} test segmentations created\r",
                end="",
            )

    def get_results_dir(self, proposed_dir: str) -> str:
        return os.path.join(proposed_dir, self.config.target)
