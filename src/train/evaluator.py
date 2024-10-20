from math import nan
from typing import Callable, Literal, Optional

from src.datasets.base_dataset import Batch
from src.models.base_model import Loss, ModelOutput
from src.train.history import DecodedPredictionBatch, MetricEntry, SingleEpochHistory


class Evaluator:
    def __init__(
        self,
        mode: Literal["train", "val", "test"],
        track_non_test_predictions: bool = False,
        extract_target_labels: Optional[Callable[[Batch], list[str]]] = None,
        extract_predicted_labels: Optional[Callable[[ModelOutput], list[str]]] = None,
    ):
        self.running_loss = 0.0
        self.n_losses = 0
        self.latest_loss = nan
        self.mode = mode
        self.track_non_test_predictions = track_non_test_predictions
        self.history = SingleEpochHistory()
        self.extract_target_labels = extract_target_labels
        self.extract_predicted_labels = extract_predicted_labels

    def track_batch(self, predictions: ModelOutput, loss: Loss, sample: Batch):
        assert loss.loss is not None
        self.running_loss += loss.loss.item()
        self.n_losses += 1
        self.latest_loss = loss.loss.item()
        decoded = (
            DecodedPredictionBatch(
                self.extract_predicted_labels(predictions),
                self.extract_target_labels(sample),
            )
            if self.extract_predicted_labels
            and self.extract_target_labels
            and sample.target is not None
            else None
        )
        self.history.add_batch_metric(
            MetricEntry(
                loss.metrics if loss.metrics is not None else {},
                loss.loss.detach().cpu().item(),
            ),
            decoded,
        )

    def get_running_loss(self):
        return self.running_loss / self.n_losses

    def get_latest_loss(self):
        return self.latest_loss

    def evaluate(self) -> SingleEpochHistory:
        return self.history

    def clean_up(self):
        pass
