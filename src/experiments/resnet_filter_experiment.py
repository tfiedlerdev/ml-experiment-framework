from typing import Literal, Any
import torch
from torch.optim.optimizer import Optimizer
from src.experiments.base_experiment import BaseExperiment, BaseExperimentArgs
from src.models.base_model import BaseModel
from src.args.yaml_config import YamlConfigModel
from src.datasets.base_dataset import BaseDataset
from src.optimizers.adam import create_adam_optimizer, AdamArgs
from src.schedulers.step_lr import StepLRArgs, create_steplr_scheduler
from src.models.resnet_filter_model import ResnetFilterModel, ResnetFilterModelArgs
from src.datasets.filter_dataset import FilterDataset, FilterDatasetArgs


class ResnetFilterExperimentArgs(
    BaseExperimentArgs,
    AdamArgs,
    StepLRArgs,
    ResnetFilterModelArgs,
    FilterDatasetArgs,
):
    do_different: bool = False


class ResnetFilterExperiment(BaseExperiment):
    def __init__(self, config: dict[str, Any], yaml_config: YamlConfigModel):
        self.config = ResnetFilterExperimentArgs(**config)
        self.resnet_filter_data = FilterDataset(self.config, yaml_config)
        super().__init__(config, yaml_config)

    def get_name(self) -> str:
        return "resnet_filter_experiment"

    def _create_dataset(
        self, split: Literal["train", "val", "test"] = "train"
    ) -> BaseDataset:
        return self.resnet_filter_data.get_split(split)

    def _create_model(self) -> BaseModel:
        return ResnetFilterModel(self.config)

    @classmethod
    def get_args_model(cls):
        return ResnetFilterExperimentArgs

    def create_optimizer(self) -> Optimizer:
        return create_adam_optimizer(self.model, self.config)

    def create_scheduler(
        self, optimizer: Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        return create_steplr_scheduler(optimizer, self.config)

    def get_loss_name(self) -> str:
        return "bce_loss"
