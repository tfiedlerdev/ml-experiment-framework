import os
import matplotlib.pyplot as plt
from typing import Literal, Any
import torch
from torch.optim.optimizer import Optimizer
from src.datasets.ukbiobank_dataset import UkBiobankDataset, UkBiobankDatasetArgs
from src.experiments.base_experiment import BaseExperiment, BaseExperimentArgs
from src.models.base_model import BaseModel
from src.args.yaml_config import YamlConfigModel
from src.datasets.base_dataset import BaseDataset
from src.optimizers.adam import create_adam_optimizer, AdamArgs
from src.schedulers.step_lr import StepLRArgs, create_steplr_scheduler
from src.models.resnet_filter_model import ResnetFilterModel, ResnetFilterModelArgs
from src.datasets.filter_dataset import (
    FilterDataset,
    FilterDatasetArgs,
    FilterFileReference,
)
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

import seaborn as sns


class ResnetFilterExperimentArgs(
    BaseExperimentArgs,
    AdamArgs,
    StepLRArgs,
    ResnetFilterModelArgs,
    FilterDatasetArgs,
):
    apply_to_uk_bio_bank: bool = False


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

    def run_after_training(self, trained_model: BaseModel):
        self.model.eval()
        test_loader = self._create_dataloader(split="test")
        self.create_cm(trained_model, test_loader, self.results_dir)
        if self.config.apply_to_uk_bio_bank:
            self._run_on_uk_biobank(trained_model)

    def _run_on_uk_biobank(self, trained_model: BaseModel):
        args = UkBiobankDatasetArgs(
            train_percentage=1.0,
        )
        bio_bank_data = UkBiobankDataset(args, self.yaml_config).get_split("train")
        data = FilterDataset(
            self.config,
            self.yaml_config,
            samples=[
                FilterFileReference(str(p.img_path), 0, True)
                for p in bio_bank_data.samples
            ],
        )
        all_samples_loader = DataLoader(
            data,
            batch_size=self.base_config.batch_size,
            shuffle=True,
            collate_fn=data.get_collate_fn(),
        )

        os.makedirs(f"{self.results_dir}/ukbiobank", exist_ok=True)
        predictions_with_probs = []
        with torch.no_grad():
            for i, data in enumerate(all_samples_loader):
                print(f"processing batch {i}/{len(all_samples_loader)}", end="\r")

                outputs = trained_model(data.cuda())
                preds = torch.argmax(outputs.logits, dim=1)

                for i, filepath in enumerate(data.file_paths):
                    pred = preds[i]
                    prob = outputs.logits[i].softmax(0)

                    predictions_with_probs.append((filepath, prob[0].item(), prob[1].item(), pred.item()))
        
        with open(f"{self.results_dir}/ukbiobank/predictions.csv", "w") as f:
            f.write("file_path,neg_prob,pos_prob,prediction\n")
            for p in predictions_with_probs:
                f.write(f"{p[0]},{p[1]},{p[2]},{p[3]}\n")


    def create_cm(self, trained_model, data_loader, output_dir):
        all_preds = []
        all_labels = []

        fps = []
        fns = []

        os.makedirs(output_dir, exist_ok=True)

        with torch.no_grad():
            for i, data in enumerate(data_loader):
                print(f"processing batch {i}/{len(data_loader)}", end="\r")

                outputs = trained_model(data.cuda())
                preds = torch.argmax(outputs.logits, dim=1)

                for i, filepath in enumerate(data.file_paths):
                    pred = preds[i]
                    prob = outputs.logits[i].softmax(0)
                    label = data.target[i].argmax(0)

                    if pred == 1 and label == 0:
                        fps.append((filepath, prob[0].item(), prob[1].item()))
                    elif pred == 0 and label == 1:
                        fns.append((filepath, prob[0].item(), prob[1].item()))

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(data.target.argmax(1).cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["0", "1"],
            yticklabels=["0", "1"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix, {len(all_preds)} total samples")
        plt.savefig(f"{output_dir}/confusion_matrix.png")

        with open(f"{output_dir}/fps.csv", "w") as f:
            f.write("file_path,neg_prob,pos_prob\n")
            for fp in fps:
                f.write(f"{fp[0]},{fp[1]},{fp[2]}\n")
        with open(f"{output_dir}/fns.csv", "w") as f:
            f.write("file_path,neg_prob,pos_prob\n")
            for fn in fns:
                f.write(f"{fn[0]},{fn[1]},{fn[2]}\n")

