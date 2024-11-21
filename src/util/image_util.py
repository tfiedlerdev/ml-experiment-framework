from typing import cast
import cv2
from git import Optional
import torch
import pickle
import os
from tqdm import tqdm


def calculate_rgb_mean_std(img_paths: list[str], cache_path: Optional[str] = None):
    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return cast(
                tuple[tuple[float, float, float], tuple[float, float, float]],
                pickle.load(f),
            )

    r_mean = g_mean = b_mean = 0
    r_std = g_std = b_std = 0
    for path in tqdm(img_paths):
        img = torch.tensor(
            cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB),
            dtype=torch.float32,
        ).transpose(0, -1)

        r_mean += img[0].mean().item()
        g_mean += img[1].mean().item()
        b_mean += img[2].mean().item()
        r_std += img[0].std().item()
        g_std += img[1].std().item()
        b_std += img[2].std().item()

    n = len(img_paths)
    r_mean /= n
    g_mean /= n
    b_mean /= n
    r_std /= n
    g_std /= n
    b_std /= n
    result = (r_mean, g_mean, b_mean), (r_std, g_std, b_std)
    if cache_path is not None:
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
    return result
