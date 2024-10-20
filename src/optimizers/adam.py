from pydantic import BaseModel
from torch.nn import Module
import torch


class AdamArgs(BaseModel):
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    eps: float = 1e-8


def create_adam_optimizer(model: Module, config: AdamArgs):
    return torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=config.eps,
    )
