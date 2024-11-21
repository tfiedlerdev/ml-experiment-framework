from src.datasets.ukbiobank_dataset import UkBiobankDatasetArgs
from src.experiments.multi_ds_vessel_experiment import MultiDSVesselExperimentArgs
from src.args.yaml_config import YamlConfig
from src.models.auto_sam_model import AutoSamModel
from torch.utils.data import DataLoader
from src.datasets.ukbiobank_dataset import UkBiobankDataset
from typing import cast
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


from src.datasets.base_dataset import Batch

import torch

# Mask output path
iteration = 0
output_path = f"/dhc/dsets/retina_masks/v{iteration}"

# Setup configs
yaml_config = YamlConfig().config
experiment_config = MultiDSVesselExperimentArgs(
    sam_model='vit_b', 
    experiment_id='label_generation', 
    prompt_encoder_checkpoint='/dhc/groups/mp2024cl2/results/drive_experiment/2024-11-15_11#24#47/prompt_encoder.pt',
    sam_checkpoint='/dhc/groups/mp2024cl2/sam_vit_b.pth',
    hard_net_cp='/dhc/groups/mp2024cl2/hardnet68.pth',
    batch_size=1,
)
ds_config = UkBiobankDatasetArgs()

# Load pretrained model
model = AutoSamModel(experiment_config)
if experiment_config.prompt_encoder_checkpoint is not None:
    print(
        f"loading prompt-encoder model from checkpoint {experiment_config.prompt_encoder_checkpoint}"
    )
    model.prompt_encoder.load_state_dict(
        torch.load(experiment_config.prompt_encoder_checkpoint, map_location="cuda"),
        strict=True,
    )
model.to("cuda")

import cv2
from pathlib import Path

# Create output folder if it does not exist
Path(output_path).mkdir(parents=True, exist_ok=True)

filtered_samples_filepath = ds_config.filter_scores_filepath

with open(filtered_samples_filepath, 'r') as f:
    for sample in f.readlines()[1:]:
        file_p, neg_prob, pos_prob, label = sample.strip().split(',')
        
        if label == '1':
            mask_out_path = f"{output_path}/{file_p.split('/')[-1]}"
            image, mask  = model.segment_image_from_file(file_p)
            mask[mask > 255 / 2] = 255
            mask[mask <= 255 / 2] = 0
            mask_img = mask * np.array([1, 1, 1])
            cv2.imwrite(mask_out_path, mask_img)
        