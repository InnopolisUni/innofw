import difflib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import DirectoryPath
from tqdm import tqdm

from innofw.utils.data_utils.preprocessing.gis_utils.cut_to_tiles import (
    cut2tiles_file,
)
from innofw.utils.data_utils.preprocessing.gis_utils.rasterize import (
    rasterize_file,
)

#
#


def find_most_similar_file(target_file):
    """function to find files which have similar names

    returns a file
    """
    candidate_files = list(target_file.parent.iterdir())

    if len(candidate_files) == 0:
        raise ValueError(f"such file: {target_file} is not found")

    sorted_candidates = sorted(
        candidate_files,
        key=lambda x: difflib.SequenceMatcher(
            None, str(x.name), str(target_file.name)
        ).ratio(),
    )
    return sorted_candidates[-1]


# DirectoryPath
# todo: create new pydantic type
# for folders to be created
def process_train_data(
    path: DirectoryPath, interim_path, dst_path, tile_step=2048
):
    """Processing of train data"""
    raster_path = path / "raster"
    geom_path = path / "razmetka"

    def find_geom_files(files, geom_path):
        files = [
            geom_path / file.parent.name / file.stem / f"{file.stem}.shp"
            for file in files
        ]
        files = [x if x.exists() else find_most_similar_file(x) for x in files]
        return files

    raster_files = list(raster_path.rglob("*.tif"))
    # geom_files = list(geom_path.rglob("*.shp"))
    geom_files = find_geom_files(raster_files, geom_path)

    assert all([x for x in geom_files if x.exists()])  # code tester # quality

    assert len(raster_files) == len(geom_files)

    # convert geom to raster files
    # split raster files to bands

    # assert that all files have different file.stem

    # create a folder for images
    dst_img_path: Path = dst_path / "img"
    dst_mask_path: Path = dst_path / "mask"

    dst_img_path.mkdir(exist_ok=True, parents=True)
    dst_mask_path.mkdir(exist_ok=True, parents=True)

    for indx, (raster_file, geom_file) in tqdm(
        enumerate(zip(raster_files, geom_files))
    ):
        # import logging
        # logger = logging.getLogger(__file__)
        # logger.setLevel(logging.INFO)

        # logging.info(raster_file.stem)     # parent.

        save_path: Path = interim_path / raster_file.stem
        # create a folder in the interim folder
        save_path.mkdir(exist_ok=True, parents=True)

        # copy raster file
        _dst_path = save_path / raster_file.name
        if not _dst_path.exists():
            shutil.copyfile(raster_file, _dst_path)

        # rasterize geom and save it in this folder
        # todo: add multiprocessing # todo: what about asyncio? # todo: multithreading?
        mask_name = "mask.tif"
        mask_file = save_path / mask_name

        if not mask_file.exists():
            rasterize_file(geom_file, raster_file, mask_file)

        cut2tiles_file(
            raster_file,
            tile_size=tile_size,
            enumerate_start=0,
            dst_path=dst_img_path,
            suffix=f"{indx}_",
        )

        cut2tiles_file(
            mask_file,
            tile_size=tile_size,
            enumerate_start=0,
            dst_path=dst_mask_path,
            suffix=f"{indx}_",
        )


def prepare_data(
    src_path, dst_path, interim_path: Optional[Path] = None, flush_interim=True
) -> Path:
    if interim_path is None:
        # get temp path
        raise NotImplementedError()

    train_data = src_path / "training"
    process_train_data(train_data, interim_path / "train", dst_path / "train")

    test_data = src_path / "test"
    raise NotImplementedError()
    return Path(".")  # todo: fix this


if __name__ == "__main__":
    src_path = Path(
        "/home/qazybek/NVME/data_n_weights/linear-road-bin-seg-oftp/200323/raw"
    )

    date = datetime.today().strftime("%Y-%m-%d")

    interim_path = Path(
        "/home/qazybek/NVME/data_n_weights/linear-road-bin-seg-oftp/200323/interim",
        date,
    )

    tile_size = 2048
    reprojected = False
    class_balancing = False  # True

    if class_balancing:
        raise NotImplementedError()

    if reprojected:
        raise NotImplementedError()

    dst_path = Path(
        "/home/qazybek/NVME/data_n_weights/linear-road-bin-seg-oftp/200323/processed/",
        f"{date}-{tile_size}",
    )

    interim_path = prepare_data(
        src_path, dst_path, interim_path, flush_interim=False
    )

# tasks:
# compute weights
# find a way to combine two datasets
# use a pretrained model

# optional:
# proper logging, for god's sake
# proper inference pipeline, for god's sake
# prep test data
