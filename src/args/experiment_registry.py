from typing import Type
from src.experiments.retina_experiment import RetinaExperiment
from src.experiments.mnist_experiment import MnistExperiment
from src.experiments.base_experiment import BaseExperiment

experiments: dict[str, Type[BaseExperiment]] = {
    "mnist": MnistExperiment,
    "retina": RetinaExperiment,
}
