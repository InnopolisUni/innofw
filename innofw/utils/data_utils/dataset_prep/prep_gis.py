import os
from pathlib import Path

import pandas as pd
from fire import Fire
from PIL import Image
from pydantic import DirectoryPath, validate_arguments
from torchvision import transforms


@validate_arguments
def prep(img_dir: DirectoryPath, boundary_dir: DirectoryPath, out_dir: Path):
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
    densities["weights"] = (1 - densities["file_weights"]) * (
        1 - densities["pix_weight"]
    )
    print(densities.sample(n=50))
    print(len(densities))

    # Save densities to .csv
    csv_file = "arable_weights_filtered_tile_1600_040923.csv"
    densities.to_csv(out_dir / csv_file)
    print(f"Weights saved to {csv_file}")


if __name__ == "__main__":
    Fire(prep)
