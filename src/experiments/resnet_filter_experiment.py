import matplotlib.pyplot as plt
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
from sklearn.metrics import confusion_matrix
import seaborn as sns


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

    def run_after_training(self, trained_model: BaseModel):
        self.model.eval()
        test_loader = self._create_dataloader(split="test")

        all_preds = []
        all_labels = []

        fps = []
        fns = []

        with torch.no_grad():
            for data in test_loader:
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
        plt.savefig(self.results_dir + "/confusion_matrix.png")

        with open(self.results_dir + "/fps.csv", "w") as f:
            f.write("file_path,neg_prob,pos_prob\n")
            for fp in fps:
                f.write(f"{fp[0]},{fp[1]},{fp[2]}\n")
        with open(self.results_dir + "/fns.csv", "w") as f:
            f.write("file_path,neg_prob,pos_prob\n")
            for fn in fns:
                f.write(f"{fn[0]},{fn[1]},{fn[2]}\n")
