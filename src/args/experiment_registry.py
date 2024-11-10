from typing import Type
from src.experiments.multi_ds_vessel_experiment import MultiDsVesselExperiment
from src.experiments.hrf_experiment import HrfExperiment
from src.experiments.drive_experiment import DriveExperiment
from src.experiments.refuge_experiment import RefugeExperiment
from src.experiments.mnist_experiment import MnistExperiment
from src.experiments.base_experiment import BaseExperiment

experiments: dict[str, Type[BaseExperiment]] = {
    "mnist": MnistExperiment,
    "refuge": RefugeExperiment,
    "drive": DriveExperiment,
    "multi_ds_vessel": MultiDsVesselExperiment,
    "hrf": HrfExperiment,
}
