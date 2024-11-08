from typing import Literal, Any
import torch
from torch.optim.optimizer import Optimizer
from src.datasets.retina_dataset import RetinaDataset, RetinaDatasetArgs
from src.models.auto_sam_model import AutoSamModel, AutoSamModelArgs
from src.experiments.base_experiment import BaseExperiment, BaseExperimentArgs
from src.models.base_model import BaseModel
from src.args.yaml_config import YamlConfigModel
from src.datasets.base_dataset import BaseDataset
from src.optimizers.adam import create_adam_optimizer, AdamArgs
from src.schedulers.step_lr import StepLRArgs, create_steplr_scheduler


class RetinaExperimentArgs(
    BaseExperimentArgs, AdamArgs, StepLRArgs, RetinaDatasetArgs, AutoSamModelArgs
):

    pass


class RetinaExperiment(BaseExperiment):
    def __init__(self, config: dict[str, Any], yaml_config: YamlConfigModel):
        self.config = RetinaExperimentArgs(**config)
        self.retina_data = RetinaDataset(config=self.config, yaml_config=yaml_config)
        super().__init__(config, yaml_config)

    def get_name(self) -> str:
        return "retina_experiment"

    def _create_dataset(
        self, split: Literal["train", "val", "test"] = "train"
    ) -> BaseDataset:
        return self.retina_data.get_split(split)

    def _create_model(self) -> BaseModel:
        return AutoSamModel(self.config)

    @classmethod
    def get_args_model(cls):
        return RetinaExperimentArgs

    def create_optimizer(self) -> Optimizer:
        return create_adam_optimizer(self.model, self.config)

    def create_scheduler(
        self, optimizer: Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        return create_steplr_scheduler(optimizer, self.config)

    def get_loss_name(self) -> str:
        return "dice+bce"
