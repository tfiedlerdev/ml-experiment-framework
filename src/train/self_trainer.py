import os
import uuid
from typing import Literal, cast

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from src.experiments.self_learning_experiment import SelfLearningExperiment
from src.datasets.base_dataset import Batch
from src.experiments.base_experiment import BaseExperiment
from src.train.evaluator import Evaluator
from src.train.history import EpochLosses, SingleEpochHistory, TrainHistory
from src.train.trainer import Trainer


class SelfTrainer(Trainer):
    def __init__(self, experiment: SelfLearningExperiment):
        self.cfg = experiment.config
        super().__init__(experiment)
        self.student_model = experiment.student_model

    def _train_epoch(self, data_loader: DataLoader):
        self.model.eval()
        self.student_model.train()
        evaluator = self.experiment.create_evaluator("train")

        for i, batch in enumerate(data_loader):
            batch = cast(Batch, batch).to(self.device)

            # Create pseudo labels for unlabeled data
            # TODO: Input is already augmented here
            # Does input augmentation reduce the quality of the pseudo labels?
            with torch.no_grad():
                batch.target = self.model.forward(batch).logits

            self.optimizer.zero_grad()

            if self.config.whiteNoiseSD > 0:
                input = batch.input
                noised_input = input + (
                    torch.randn(input.shape, device=input.device)
                    * self.config.whiteNoiseSD
                )
                batch.input = noised_input

            if self.config.constantOffsetSD > 0:
                input = batch.input
                offset_input = input + (
                    torch.randn(
                        [input.shape[0], 1, input.shape[2]], device=input.device
                    )
                    * self.config.constantOffsetSD
                )
                batch.input = offset_input

            # Make predictions for this batch
            with torch.enable_grad():
                # calculate gradient for whole model (but only optimize parts)
                outputs = self.student_model.forward(batch)

            loss = self.student_model.compute_loss(outputs, batch)
            loss.loss.backward()

            if self.config.gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(  # type: ignore
                    self.student_model.parameters(), self.config.gradient_clipping
                )

            # Adjust weights of teacher model
            with torch.no_grad():  # Disable gradient tracking
                for teacher_param, student_param in zip(
                    self.model.parameters(), self.student_model.parameters()
                ):
                    teacher_param.data = (
                        (1 - self.cfg.teacher_student_update_ratio) * teacher_param.data
                        + self.cfg.teacher_student_update_ratio * student_param.data
                    )

            # Adjust learning weights
            self.optimizer.step()
            evaluator.track_batch(outputs, loss, batch)
            if (
                i % self.config.log_every_n_batches
                == self.config.log_every_n_batches - 1
            ):
                self._log_intermediate(i, len(self.dataloader_train), evaluator)
        results = evaluator.evaluate()
        evaluator.clean_up()
        return results

    def train(self):
        history: list[EpochLosses] = (
            self.experiment.checkpoint_history.epochs
            if self.experiment.checkpoint_history is not None
            else []
        )
        best_model_val_metric = float(
            "inf" if self.config.minimize_best_model_metric else "-inf"
        )
        best_model_path = os.path.join(
            self.yaml_config.cache_dir,
            "model_checkpoints",
            str(uuid.uuid4()),
            "best_model.pt",
        )
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

        def get_relevant_metric(epoch_hist: SingleEpochHistory):
            return (
                epoch_hist.get_average().loss
                if self.config.best_model_metric == "loss"
                else epoch_hist.get_average().metrics[self.config.best_model_metric]
            )

        best_model_epoch = -1

        for epoch in range(self.config.epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")

            train_losses = self._train_epoch(self.dataloader_train)
            val_losses = self.evaluate_epoch("val")
            self.scheduler.step()

            print(
                f"\n\n{'='*20}\nFinished Epoch {epoch + 1}/{self.config.epochs} "
                f"train {self.loss_name}-loss: {train_losses.get_average().loss} "
                f"val {self.loss_name}-loss: {val_losses.get_average().loss}"
            )
            epoch_losses = EpochLosses(train_losses, val_losses)
            history.append(epoch_losses)
            self._log_epoch_wandb(epoch_losses)
            if self.config.return_best_model:
                curr_epoch_val_metric = get_relevant_metric(val_losses)

                is_better = (
                    curr_epoch_val_metric < best_model_val_metric
                    if self.config.minimize_best_model_metric
                    else curr_epoch_val_metric > best_model_val_metric
                )
                if is_better:
                    best_model_val_metric = curr_epoch_val_metric
                    torch.save(self.model.state_dict(), best_model_path)
                    print(f"\n\nSaving model checkpoint at {best_model_path}\n")
                    best_model_epoch = epoch

            if (
                self.config.early_stopping_patience is not None
                and len(history) >= self.config.early_stopping_patience
            ):
                relevant_metric_history = [
                    get_relevant_metric(epoch_loss.val_losses) for epoch_loss in history
                ][-self.config.early_stopping_patience :]

                # Adapt basline metric via early_stopping_epsilon
                if self.config.minimize_best_model_metric:
                    relevant_metric_history[0] -= self.config.early_stopping_delta
                else:
                    relevant_metric_history[0] += self.config.early_stopping_delta
                best_index = (
                    np.argmin(relevant_metric_history)
                    if self.config.minimize_best_model_metric
                    else np.argmax(relevant_metric_history)
                )
                if best_index == 0:
                    print(
                        f"\nEarly stopping after {epoch} epochs ({self.config.early_stopping_patience} epochs without improvement in validation {self.config.best_model_metric} metrics)"
                    )
                    break

        if self.config.return_best_model:
            self.model.load_state_dict(torch.load(best_model_path))
            os.remove(best_model_path)
            os.rmdir(os.path.dirname(best_model_path))
            print("Loaded model with best validation loss of this experiment from disk")

        test_losses = self.evaluate_epoch("test")
        wandb.log(self._get_wandb_metrics(test_losses, "test"))
        print(f"\nTest loss ({self.loss_name}): {test_losses.get_average().loss}")
        return self.model, TrainHistory(history, test_losses, best_model_epoch)
