import os
from src.datasets.ukbiobank_dataset import UkBiobankDatasetArgs
from src.args.yaml_config import YamlConfig
from src.models.auto_sam_model import AutoSamModel, AutoSamModelArgs
from tqdm import tqdm
import numpy as np
import torch
from src.args.yaml_config import YamlConfig
from pathlib import Path

yaml_config = YamlConfig().config
# Mask output path
iteration = 0
output_path = Path(yaml_config.ukbiobank_data_dir) / f"v{iteration}"
prompt_encoder_checkpoint = "/dhc/groups/mp2024cl2/results/multi_ds_vessel_experiment/2024-11-21_14#26#29/prompt_encoder.pt"
# Setup configs
yaml_config = YamlConfig().config
sam_config = AutoSamModelArgs(
    sam_model="vit_b",
    sam_checkpoint="/dhc/groups/mp2024cl2/sam_vit_b.pth",
    hard_net_cp="/dhc/groups/mp2024cl2/hardnet68.pth",
)
ds_config = UkBiobankDatasetArgs()

# Load pretrained model
model = AutoSamModel(sam_config)
if prompt_encoder_checkpoint is not None:
    print(f"loading prompt-encoder model from checkpoint {prompt_encoder_checkpoint}")
    model.prompt_encoder.load_state_dict(
        torch.load(prompt_encoder_checkpoint, map_location="cuda"),
        strict=True,
    )
model.to("cuda")

import cv2
import json
from pathlib import Path

# Create output folder if it does not exist
Path(output_path).mkdir(parents=True, exist_ok=True)

filtered_samples_filepath = ds_config.filter_scores_filepath
metadata = {
    **sam_config.model_dump(),
    "prompt_encoder_checkpoint": prompt_encoder_checkpoint,
}

with open(os.path.join(output_path, "metadata.json"), "w") as f:
    json.dump(metadata, f)

with open(filtered_samples_filepath, "r") as f:
    samples = [line.strip().split(",") for line in f.readlines()[1:]]

relevant_filepaths = [
    file_p for file_p, neg_prob, pos_prob, label in samples if label == "1"
]

for p in tqdm(relevant_filepaths, miniters=500):
    mask_out_path = f"{output_path}/{os.path.basename(p)}"
    image, mask = model.segment_image_from_file(p)
    mask[mask > 255 / 2] = 255
    mask[mask <= 255 / 2] = 0
    mask_img = mask * np.array([1, 1, 1])
    cv2.imwrite(mask_out_path, mask_img)

print("Done generating masks. Saved in ", output_path)
