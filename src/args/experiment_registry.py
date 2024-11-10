from typing import Type
from src.experiments.refuge_experiment import RefugeExperiment
from src.experiments.mnist_experiment import MnistExperiment
from src.experiments.base_experiment import BaseExperiment

experiments: dict[str, Type[BaseExperiment]] = {
    "mnist": MnistExperiment,
    "retina": RefugeExperiment,
}
