from typing import Callable, Literal, Optional
from typing_extensions import Self
from pydantic import BaseModel, Field
from src.models.auto_sam_model import SAMSampleFileReference
from src.args.yaml_config import YamlConfigModel
from src.datasets.base_dataset import BaseDataset, JoinedDataset
from src.datasets.chasedb1_dataset import ChaseDb1Dataset, ChaseDb1DatasetArgs
from src.datasets.drive_dataset import DriveDataset, DriveDatasetArgs
from src.datasets.hrf_dataset import HrfDataset, HrfDatasetArgs
from src.datasets.stare_dataset import STAREDataset, STAREDatasetArgs


class JoinedRetinaDatasetArgs(
    DriveDatasetArgs, ChaseDb1DatasetArgs, STAREDatasetArgs, HrfDatasetArgs
):
    pass


class JoinedRetinaDataset(JoinedDataset):
    def __init__(self, datasets: list[BaseDataset], collate: Optional[Callable] = None):
        super().__init__(datasets, collate)  # type: ignore

    def get_file_refs(self) -> list[SAMSampleFileReference]:
        return [sample for ds in self.datasets for sample in ds.samples]  # type: ignore

    @classmethod
    def from_config(cls, config: JoinedRetinaDatasetArgs, yaml_config: YamlConfigModel):
        drive = DriveDataset(config=config, yaml_config=yaml_config)
        chase_db1 = ChaseDb1Dataset(config=config, yaml_config=yaml_config)
        hrf = HrfDataset(config=config, yaml_config=yaml_config)
        stare = STAREDataset(config=config, yaml_config=yaml_config)
        return cls([drive, chase_db1, hrf, stare], drive.get_collate_fn())
