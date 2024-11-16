import torch
from src.models.base_model import BaseModel, ModelOutput, Loss
from src.datasets.base_dataset import Batch
from src.util.nn_helper import create_fully_connected, ACTIVATION_FUNCTION
from pydantic import BaseModel as PDBaseModel
from torch.nn import CrossEntropyLoss


class ResnetFilterModelArgs(PDBaseModel):
    activation: ACTIVATION_FUNCTION = "gelu"


class ResnetFilterModel(BaseModel):
    resnet_version = "resnet18"
    resnet_output_size = 1000

    def __init__(self, config: ResnetFilterModelArgs):
        super().__init__()
        self.resnet = torch.hub.load(
            "pytorch/vision:v0.10.0", ResnetFilterModel.resnet_version, pretrained=True
        )
        self.model = create_fully_connected(ResnetFilterModel.resnet_output_size, 2)
        self.loss = CrossEntropyLoss()

    def forward(self, batch: Batch) -> ModelOutput:
        with torch.no_grad():
            out = self.resnet(batch.input)
        out = self.model.forward(out)
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
