#
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import rasterio as rio
from fire import Fire
from pydantic import validate_arguments
from pytorch_toolbelt.inference.tiles import ImageSlicer
from tqdm import tqdm

#


def cut2tiles_file(
    file, dst_path, tile_size, tile_step=None, enumerate_start=0, suffix=""
):
    if tile_step is None:
        tile_step = tile_size

    raster = rio.open(file)
    metadata = raster.meta
    channels_num = raster.count

    img = np.dstack([raster.read(i) for i in range(1, channels_num + 1)])
    return cut2tiles_array(
        dst_path,
        tile_size,
        tile_step,
        enumerate_start,
        suffix,
        metadata,
        channels_num,
        img,
    )


def cut2tiles_array(
    dst_path,
    tile_size,
    tile_step,
    enumerate_start,
    suffix,
    metadata,
    channels_num,
    img,
):
    tiler = ImageSlicer(img.shape, tile_size=tile_size, tile_step=tile_step)
    if channels_num == 1:
        tiles = [tile.astype(np.int8) for tile in tiler.split(img)]
    else:
        tiles = [tile for tile in tiler.split(img)]

    metadata.update(
        driver="GTiff", count=channels_num, height=tile_size, width=tile_size
    )
    dst_path.mkdir(exist_ok=True, parents=True)
    i = enumerate_start
    for i, tile in tqdm(enumerate(tiles, start=enumerate_start)):
        with rio.open(dst_path / f"{suffix}{i}.tif", "w", **metadata) as f:
            if len(tile.shape) == 3:
                tile = np.moveaxis(tile, -1, 0)
                f.write(tile)
            else:
                f.write(tile, 1)
    return i


def process_folder(
    file_num, folder, filename, dst_path, tile_size, tile_step=None
):
    if "," in filename or type(filename) == tuple:
        if "," in filename:
            names = [folder / ch for ch in filename.split(",")]
        else:
            names = filename
        try:
            img = np.dstack([rio.open(name).read(1) for name in names])
            metadata = rio.open(names[0]).meta
        except:
            try:
                img = np.dstack(
                    [
                        rio.open(folder / f"{name}.tif").read(1)
                        for name in names
                    ]
                )
                metadata = rio.open(folder / f"{names[0]}.tif").meta
            except:
                img = np.dstack(
                    [
                        rio.open(folder / f"{name}.jp2").read(1)
                        for name in names
                    ]
                )
                metadata = rio.open(folder / f"{names[0]}.jp2").meta

        cut2tiles_array(
            dst_path,
            tile_size,
            tile_step,
            enumerate_start=0,
            suffix=f"{file_num}_",
            metadata=metadata,
            channels_num=len(names),
            img=img,
        )
    if filename is None:
        name = [i for i in folder.rglob("*.tif") if "label.tif" not in i.name][
            0
        ]
    else:
        name = folder / filename

    return cut2tiles_file(
        file=name,
        dst_path=dst_path,
        tile_size=tile_size,
        tile_step=tile_step,
        enumerate_start=0,
        suffix=f"{file_num}_",
    )


def process_file(file_num, file, dst_path, tile_size, tile_step=None):
    return cut2tiles_file(
        file=file,
        dst_path=dst_path,
        tile_size=tile_size,
        tile_step=tile_step,
        enumerate_start=0,
        suffix=f"{file_num}_",
    )


@validate_arguments
def cut2tiles(
    src_path: Path,
    dst_path: Path,
    tile_size: int,
    tile_step: Optional[int] = None,
    filename: Union[Optional[Tuple], Optional[str]] = None,
):
    """
        Arguments:
            src_path - folder with rasters or folder with folder with rasters
                src_path/
                    raster.tif
                    raster2.tif
                    ...

                or 

                src_path/
                    folder/
                        raster.tif
                    folder2/
                        raster2.tif
                ...
            dst_path - folder where to store results
            tile_size - size of the tile
            filename - files with this name will be processed

        Usage:
        >>> python gis_utils/cut_to_tiles.py\
                --src_path /some/path\
                --dst_path /other/path\
                --tile_size 256
    """
    folders = src_path.iterdir()
    folders = sorted(folders, key=lambda x: x.name)

    with Pool(32) as p:
        if folders[0].is_file():
            # for files
            p.starmap(
                partial(
                    process_file,
                    dst_path=dst_path,
                    tile_size=tile_size,
                    tile_step=tile_step,
                ),
                enumerate(folders),
            )
        else:
            # for folders
            p.starmap(
                partial(
                    process_folder,
                    filename=filename,
                    dst_path=dst_path,
                    tile_size=tile_size,
                    tile_step=tile_step,
                ),
                enumerate(folders),
            )


if __name__ == "__main__":
    Fire(cut2tiles)
