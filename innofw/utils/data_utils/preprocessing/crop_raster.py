#
import random
from pathlib import Path

import fire
import rasterio
from rasterio.windows import Window

from innofw.constants import PathLike

#


def crop_raster(src_path: Path, dst_path: Path, height=512, width=512):
    with rasterio.open(src_path) as src:
        # Generate a random window origin (upper left) that ensures the window
        # doesn't go outside the image. i.e. origin can only be between
        # 0 and image width or height less the window width or height
        xmin, xmax = 0, src.width - width
        ymin, ymax = 0, src.height - height
        xoff, yoff = random.randint(xmin, xmax), random.randint(ymin, ymax)

        # Create a Window and calculate the transform from the source dataset
        window = Window(xoff, yoff, width, height)
        transform = src.window_transform(window)

        # Create a new cropped raster to write to
        profile = src.profile
        profile.update(
            {"height": height, "width": width, "transform": transform}
        )

        with rasterio.open(dst_path, "w+", **profile) as dst:
            # Read the data from the window and write it to the output raster
            dst.write(src.read(window=window))


def crop_multiple_rasters(
    src_folder_path: PathLike,
    dst_folder_path: PathLike,
    height: int = 100,
    width: int = 100,
    extension: str = ".tif",
):
    files = Path(src_folder_path).rglob(f"*{extension}")
    dst_folder_path = Path(dst_folder_path)
    dst_folder_path.mkdir(exist_ok=True, parents=True)
    for file in files:
        crop_raster(file, dst_folder_path / file.name, height, width)


if __name__ == "__main__":
    fire.Fire(crop_multiple_rasters)
