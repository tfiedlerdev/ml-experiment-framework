from typing import Literal, Any
import torch
from torch.optim.optimizer import Optimizer
from src.experiments.base_experiment import BaseExperiment, BaseExperimentArgs
from src.models.base_model import BaseModel
from src.args.yaml_config import YamlConfigModel
from src.datasets.base_dataset import BaseDataset
from src.optimizers.adam import create_adam_optimizer, AdamArgs
from src.schedulers.step_lr import StepLRArgs, create_steplr_scheduler
from src.models.mnist_fc_model import MnistFcModel, MnistFcModelArgs
from torchvision.datasets import MNIST
import os
from src.datasets.mnist_dataset import MnistDataset, MnistDatasetArgs


class MnistExperimentArgs(
    BaseExperimentArgs, AdamArgs, StepLRArgs, MnistFcModelArgs, MnistDatasetArgs
):
    do_different: bool = False


class MnistExperiment(BaseExperiment):
    def __init__(self, config: dict[str, Any], yaml_config: YamlConfigModel):
        self.config = MnistExperimentArgs(**config)
        self.mnist_data = MnistDataset(self.config, yaml_config)
        super().__init__(config, yaml_config)

    def get_name(self) -> str:
        return "mnist_experiment"

    def _create_dataset(
        self, split: Literal["train", "val", "test"] = "train"
    ) -> BaseDataset:
        return self.mnist_data.get_split(split)

    def _create_model(self) -> BaseModel:
        return MnistFcModel(self.config)

    @classmethod
    def get_args_model(cls):
        return MnistExperimentArgs

    def create_optimizer(self) -> Optimizer:
        return create_adam_optimizer(self.model, self.config)

    def create_scheduler(
        self, optimizer: Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        return create_steplr_scheduler(optimizer, self.config)

    def get_loss_name(self) -> str:
        return "cross_entropy"
