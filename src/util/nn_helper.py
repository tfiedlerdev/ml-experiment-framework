from typing import Literal
from torch.nn import Linear, BatchNorm1d
from torch import nn

ACTIVATION_FUNCTION = Literal[
    "gelu",
    "identity",
    "relu",
    "sigmoid",
    "tanh",
]

ACT2FN: dict[ACTIVATION_FUNCTION, nn.Module] = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "identity": nn.Identity()
}


def create_fully_connected(
    input_size: int,
    output_size: int,
    hidden_sizes=[],
    activation: ACTIVATION_FUNCTION = "gelu",
    use_batch_norm: bool = False,
):
    classifier_layers = []
    for i in range(-1, len(hidden_sizes)):
        is_last = i + 1 == len(hidden_sizes)
        is_first = i == -1
        in_size = input_size if is_first else hidden_sizes[i]
        out_size = output_size if is_last else hidden_sizes[i + 1]
        classifier_layers.append(Linear(in_size, out_size))
        if not is_last:
            if use_batch_norm:
                classifier_layers.append(BatchNorm1d(num_features=1))
            classifier_layers.append(ACT2FN[activation])
    return nn.Sequential(*classifier_layers)