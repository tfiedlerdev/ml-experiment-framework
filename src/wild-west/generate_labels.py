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
output_path = Path(yaml_config.ukbiobank_masks_dir) / f"v{iteration}"
model_checkpoint = "/dhc/groups/mp2024cl2/results/multi_ds_vessel_experiment/reference_flashattn_waria/model.pt"
filter_threshold = yaml_config.filter_threshold
# Setup configs
sam_config = AutoSamModelArgs(
    sam_model="vit_b",
)
ds_config = UkBiobankDatasetArgs()

# Load pretrained model
model = AutoSamModel(sam_config)
if model_checkpoint is not None:
    print(f"loading model from checkpoint {model_checkpoint}")
    model.load_state_dict(
        torch.load(model_checkpoint, map_location="cuda"),
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
    "model_checkpoint": model_checkpoint,
    "script_source": Path(__file__).read_text(),
    "filter_threshold": filter_threshold,
}


with open(os.path.join(output_path, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=5)

with open(filtered_samples_filepath, "r") as f:
    samples = [line.strip().split(",") for line in f.readlines()[1:]]

relevant_filepaths = [
    file_p
    for file_p, neg_prob, pos_prob, predicted_label in samples
    if float(pos_prob) >= filter_threshold
]
masks_out_dir = Path(f"{output_path}/generated_masks")
masks_out_dir.mkdir(parents=True, exist_ok=True)
for p in tqdm(relevant_filepaths, miniters=500):
    mask_out_path = masks_out_dir / os.path.basename(p)
    image, mask = model.segment_image_from_file(p)
    mask[mask > 255 / 2] = 255
    mask[mask <= 255 / 2] = 0
    mask_img = mask * np.array([1, 1, 1])
    cv2.imwrite(str(mask_out_path), mask_img)

print("Done generating masks. Saved in ", output_path)
