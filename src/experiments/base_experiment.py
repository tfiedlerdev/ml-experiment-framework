import json
import os
import sys
from abc import ABCMeta, abstractmethod
from datetime import datetime
from typing import Literal, Optional, Type, Any

import numpy as np
import torch
import wandb
from pydantic import BaseModel as PDBaseModel
from pydantic import Field
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from src.models.base_model import BaseModel
from src.args.yaml_config import YamlConfigModel
from src.datasets.base_dataset import BaseDataset
from src.train.evaluator import Evaluator
from src.train.history import SingleEpochHistory, TrainHistory


class BaseExperimentArgs(PDBaseModel):
    batch_size: int = Field(
        default=16, description="Batch size for training and validation"
    )
    epochs: int = 10
    wandb_experiment_name: str = "experiment_1"
    experiment_id: str = Field(
        description="Type identifier of experiment to run. Experiment is selected from experiment_registry"
    )
    log_every_n_batches: int = 10
    return_best_model: bool = True
    best_model_metric: str = Field(
        default="loss",
        description='The metric by which to measure the models performance. Can be "loss" for using the applied loss or any metric that is returned by the model',
    )
    minimize_best_model_metric: bool = Field(
        default=True,
        description="Specify if best_model_metric should be minimized or maximized",
    )
    use_wandb: bool = False
    from_checkpoint: Optional[str] = Field(
        default=None, description="(optional) Path to model checkpoint"
    )
    only_test: bool = Field(default=False, description="Only run test, skip training")
    predict_on_train: bool = Field(
        default=False, description="Run prediction on train set after model training"
    )
    gradient_clipping: Optional[float] = None
    whiteNoiseSD: float = 0.0
    constantOffsetSD: float = 0.0
    seed: int = 42
    early_stopping_patience: Optional[int] = Field(
        default=None,
        description="Number of epochs n to consider for early stopping. Once all n-1 last epochs did not improve compared to the -nth epoch, training is stopped.   If None, early stopping is disabled",
    )
    early_stopping_delta: float = Field(
        default=0.0001,
        description="Minimum delta of to be optimized metric that is considered as an improvement for early stopping",
    )
    results_subdir_name: Optional[str] = None
    use_cuda: bool = True


class BaseExperiment(metaclass=ABCMeta):
    def __init__(self, config: dict[str, Any], yaml_config: YamlConfigModel):
        self.base_config = BaseExperimentArgs(**config)
        torch.manual_seed(self.base_config.seed)
        np.random.seed(self.base_config.seed)
        self.yaml_config = yaml_config

        self.raw_config = config

        self.checkpoint_history = None
        proposed_results_dir = (
            os.path.join(
                yaml_config.results_dir,
                self.get_name(),
            )
            if self.base_config.results_subdir_name is None
            else os.path.join(
                yaml_config.results_dir,
                self.get_name(),
                self.base_config.results_subdir_name,
            )
        )
        self.results_dir = os.path.join(
            self.get_results_dir(proposed_results_dir),
            f"{datetime.now():%Y-%m-%d_%H#%M#%S}",
        )

        os.makedirs(self.results_dir, exist_ok=True)
        with open(os.path.join(self.results_dir, "config.json"), "w") as f:
            config_copy = dict(config)
            config_copy["repro_cmd"] = "python " + " ".join(sys.argv)
            json.dump(config_copy, f, indent=5)
        self.model = self._create_model().to(self.get_device())
        self.checkpoint_history = None
        if self.base_config.from_checkpoint is not None:
            self.load_trained_model(self.base_config.from_checkpoint)
            history_path = os.path.join(
                os.path.dirname(self.base_config.from_checkpoint), "history.json"
            )
            if os.path.exists(history_path):
                print("Attempting to load history from checkpoint")
                try:
                    self.checkpoint_history = TrainHistory.from_json(history_path)
                except Exception as e:
                    print(f"Failed to load history from checkpoint: {e}")

            print("")

    def run(self):
        from src.train.trainer import Trainer

        if self.base_config.use_wandb:
            wandb.login(key=self.yaml_config.wandb_api_key, relogin=True)

        trainer = Trainer(self)
        wandb.init(
            project=self.yaml_config.wandb_project_name,
            entity=self.yaml_config.wandb_entity,
            config=self.raw_config,
            name=self.base_config.wandb_experiment_name,
            dir=self.yaml_config.cache_dir,
            save_code=True,
            mode="online" if self.base_config.use_wandb else "disabled",
        )
        if wandb.run is None:
            raise Exception("wandb init failed. wandb.run is None")
        with wandb.run:
            if not self.base_config.only_test:
                trained_model, history = trainer.train()
                self.store_trained_model(trained_model)
                with open(os.path.join(self.results_dir, "history.json"), "w") as f:
                    json.dump(history.to_dict(), f, indent=5)

                self.plot_results(history)
                self.process_test_results(history.test_losses)
                self.run_after_training(trained_model)
            else:
                test_results = trainer.evaluate_epoch("test")
                if test_results is not None:
                    wandb.log(trainer._get_wandb_metrics(test_results, "test"))
                    self.process_test_results(test_results)
                    with open(
                        os.path.join(self.results_dir, "test_results.json"), "w"
                    ) as f:
                        json.dump(test_results.to_dict(), f, indent=5)
                self.run_after_training(self.model)
            print(f"Done. Saved results to {self.results_dir}")

    def get_results_dir(self, proposed_dir: str) -> str:
        return proposed_dir

    def run_after_training(self, trained_model: BaseModel):
        pass

    def load_trained_model(self, path: str):
        print(f"loading model from checkpoint {path}")
        self.model.load_state_dict(
            torch.load(path, map_location="cuda"),
            strict=True,
        )

    def store_trained_model(self, trained_model: torch.nn.Module):
        torch.save(
            trained_model.state_dict(),
            os.path.join(self.results_dir, "model.pt"),
        )

    def process_test_results(self, test_results: SingleEpochHistory):
        pass

    def plot_results(self, history: TrainHistory):
        history.plot(os.path.join(self.results_dir, "history.png"))

    def get_device(self):
        return "cuda" if self.base_config.use_cuda else "cpu"

    @abstractmethod
    def get_name(self) -> str:
        """Return name of experiment to determine result output dir"""
        pass

    @abstractmethod
    def _create_dataset(
        self, split: Literal["train", "val", "test"] = "train"
    ) -> BaseDataset:
        raise NotImplementedError("Implement _create_dataset in subclass")

    @abstractmethod
    def _create_model(self) -> BaseModel:
        pass

    @classmethod
    def get_args_model(cls) -> Type[BaseExperimentArgs]:
        raise NotImplementedError()

    def _create_dataloader(self, split: Literal["train", "val", "test"]) -> DataLoader:
        ds = self._create_dataset(split)
        return DataLoader(
            ds,
            batch_size=self.base_config.batch_size,
            shuffle=True,
            collate_fn=ds.get_collate_fn(),
        )

    @abstractmethod
    def create_optimizer(self) -> Optimizer:
        raise NotImplementedError()

    def create_evaluator(
        self,
        mode: Literal["train", "val", "test"],
        track_non_test_predictions: bool = False,
    ) -> Evaluator:
        return Evaluator(mode, track_non_test_predictions)

    @abstractmethod
    def create_scheduler(
        self, optimizer: Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        raise NotImplementedError()

    @abstractmethod
    def get_loss_name(self) -> str:
        """Return name of loss for logging purposes"""
        raise NotImplementedError()
