import torch
from pydantic import BaseModel
from torch.optim.optimizer import Optimizer


class StepLRArgs(BaseModel):
    step_size: int = 1
    gamma: float = 1


def create_steplr_scheduler(optimizer: Optimizer, config: StepLRArgs):
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.step_size,
        gamma=config.gamma,
    )
