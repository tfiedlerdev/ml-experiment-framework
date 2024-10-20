from src.models.base_model import BaseModel, ModelOutput, Loss
from src.datasets.base_dataset import Batch
from src.util.nn_helper import create_fully_connected, ACTIVATION_FUNCTION
from pydantic import BaseModel as PDBaseModel
from torch.nn import CrossEntropyLoss


class MnistFcModelArgs(PDBaseModel):
    hidden_sizes: list[int]
    activation: ACTIVATION_FUNCTION = "gelu"


class MnistFcModel(BaseModel):
    def __init__(self, config: MnistFcModelArgs):
        super().__init__()
        self.model = create_fully_connected(
            784, 10, config.hidden_sizes, config.activation
        )
        self.loss = CrossEntropyLoss()

    def forward(self, batch: Batch) -> ModelOutput:
        out = self.model.forward(batch.input)
        return ModelOutput(out)

    def compute_loss(self, outputs: ModelOutput, batch: Batch) -> Loss:
        assert batch.target is not None
        loss_value = self.loss.forward(outputs.logits, batch.target)
        accuracy = (outputs.logits.argmax(-1) == batch.target.argmax(-1)).sum() / len(
            batch.input
        )
        return Loss(
            loss_value,
            {"accuracy": accuracy.item(), "ce_loss": loss_value.detach().item()},
        )
