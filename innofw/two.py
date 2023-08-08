import os
import numpy as np
from PIL import Image
from torchvision import transforms

# import csv
import pandas as pd

# Directories
img_dir = "/mnt/nvmestorage/qb/data_n_weights/arable-borders-bin-seg-ndvi/200723/processed/1600/train/img"
boundary_dir = "/mnt/nvmestorage/qb/data_n_weights/arable-borders-bin-seg-ndvi/200723/processed/1600/train/boundary"

# List the image and mask files
boundary_files = sorted([f for f in os.listdir(boundary_dir) if f.endswith(".tif")])
print(len(boundary_files))

# Calculate densities
# densities = []
columns = ["file_names", "pix_weight"]

densities = pd.DataFrame(columns=columns)

for boundary_file in boundary_files:
    boundary_path = os.path.join(boundary_dir, boundary_file)
    mask = Image.open(boundary_path)
    mask_tensor = transforms.ToTensor()(mask)
    foreground_pixel_count = mask_tensor.sum().item()
    total_pixels = mask_tensor.numel()
    density = foreground_pixel_count / total_pixels

    if foreground_pixel_count / total_pixels > 0.95 or foreground_pixel_count == 0:
        # Drop samples that are too dense or too sparse
        continue

    densities = densities.append(
        pd.Series((boundary_file, density), index=columns), ignore_index=True
    )
    # densities.append((boundary_file, 1 - density))

densities["src_file"] = densities.file_names.apply(lambda x: int(x.split("_")[0]))
tile_count = densities.groupby("src_file").size()
print(tile_count)
group_weights = tile_count / len(densities)
densities["file_weights"] = densities.src_file.apply(lambda x: group_weights[x])
densities["weights"] = (1 - densities["file_weights"]) * (1 - densities["pix_weight"])

print(densities.sample(n=50))

print(len(densities))

# Save densities to .csv
csv_file = "arable_weights_filtered_tile_1600.csv"
densities.to_csv(csv_file)
print(f"Weights saved to {csv_file}")
