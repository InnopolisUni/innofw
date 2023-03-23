"""
    TODO:
        - code from this packages should be moved to another project
        - use logger
        - code should be refactored: preserve functional style but use decorators to add additional functionality
            cause in the end you want to call a function

            from gis_utils import rasterize_geom_file

            rasterize_geom_file(geom_file, dst_path)


            >>> python rasterize.py rasterize_geom_file --geom_file ... --dst_path ...
        
        - create __all__ file 
"""
from rasterio import logging

log = logging.getLogger()
log.setLevel(logging.ERROR)


# standard library
# import logging
from pathlib import Path
from typing import List, Optional, Tuple

# third party libraries
import geopandas as gpd
import affine
import numpy as np
import rasterio as rio
from rasterio import mask as msk
from fire import Fire
from pydantic import validate_arguments, FilePath
from tqdm import tqdm

# logger = logging.getLogger(__file__)
# logger.setLevel(logging.INFO)


# local modules
# from .abstract import Operation  # AbstractOperation
# from .reprojector import reproject_geom_file
# from .utils import validate_files, new_dst_path


def new_dst_path(file, src_path, dst_path):  # todo: is it needed?
    relative_path = file.relative_to(src_path)
    dst_path = (
        Path(dst_path, relative_path.parent, file.stem)
        if dst_path
        else file.parent
    )
    dst_path.mkdir(parents=True, exist_ok=True)
    return dst_path


@validate_arguments
def rasterize_file(
    geom_file: FilePath, scene_file: FilePath, new_filename: Path
) -> Path:
    """Creates rasterized file
    Output file will have same extension as scene_file
    Save dir can be omitted in that case raster will be saved in the same directory as scene file

    Args:
        geom_file: A path to the geometry file
        scene_file: A path to the reference scene file
        new_filename: A path to the raster file to be created

    Returns:
        A path to the raster file created
    """
    new_filename.parent.mkdir(exist_ok=True, parents=True)

    # logger.warning("rasterization process may require sudo rights")

    target_crs = rio.open(scene_file).crs
    shapes = get_shapes(geom_file, target_crs)
    # if there are no shapes then create a blank mask
    print(f"Number of shapes: {len(shapes)}")
    if len(shapes) == 0:
        src = rio.open(scene_file, "r")
        binary_mask = np.zeros_like(src.shape)
        meta = src.meta
    else:
        binary_mask, transform, meta = do_raster_masking(scene_file, shapes)
    meta["count"] = 1
    print(new_filename.name, binary_mask.shape)
    with rio.open(new_filename, "w", **meta) as dest:
        dest.write(binary_mask)

    # logging.info(f"Rasterized to {new_filename} {len(shapes)} shapes")

    return new_filename


def get_shapes(geom_file: Path, target_crs=None) -> List:
    """Function to get shapes from geometry file

    Args:
        geom_file: A path to the geometry file
        target_crs: A crs to which geometry file should be reprojected
    Returns:
        A list of shapes retrieved from geometry file
    """
    geom = gpd.read_file(geom_file)
    if geom.crs is None:
        geom = geom.set_crs(target_crs)
    if target_crs is not None:  # todo: should it be here?
        geom = geom.to_crs(target_crs)

    # # for multiclass case: refactor
    # try:
    #     geom = geom[geom["DN"] == 30]
    # except:
    #     pass

    shapes = geom["geometry"]
    shapes = [shape for shape in shapes if shape is not None]
    return shapes  # , geom['label']


def do_raster_masking(
    raster_file: Path, shapes: List
) -> Tuple[np.array, affine.Affine, dict]:
    """Function for raster masking

    Args:
        raster_file: A path to the raster
        shapes: A list of shapes

    Returns:
        A tuple of mask array, transformation of the mask and dictionary with metadata
    """
    with rio.open(raster_file) as src:
        img, out_transform = msk.mask(src, shapes)
        binary_img = (img > 0).astype("uint8")

        if binary_img.shape[0] != 1:  # todo: refactor this
            binary_img = binary_img[0, ...]
            binary_img = np.expand_dims(binary_img, 0)

        out_meta = src.meta.copy()
        out_meta.update({"count": 1, "driver": "GTiff"})
        out_meta["dtype"] = "uint8"
        return binary_img, out_transform, out_meta


def return_empty_mask(raster_file: Path):
    with rio.open(raster_file) as src:
        src_array = src.read(1)
        binary_img = np.zeros(src_array.shape, dtype=src_array.dtype)

        out_meta = src.meta.copy()
        out_meta.update({"count": 1, "driver": "GTiff"})
        out_meta["dtype"] = "uint8"
        return binary_img, src.transform, out_meta


def save_empty_mask(raster_file: Path, dst_filepath: Path):
    bin_img, _, meta = return_empty_mask(raster_file)

    with rio.open(dst_filepath, "w", **meta) as file:
        file.write(bin_img, 1)


def find_ref_file(folder):
    tiffs = list((folder).glob("*.tif"))
    jp2s = list((folder).glob("*.jp2"))

    return tiffs + jp2s


def rasterize_single(
    folder,
    raster_folder_path,
    ref_raster_filename=None,
    dst_folder_path=None,
    dst_file_name="label.tif",
):
    print(f"processing folder: {folder}")
    try:
        geom_file = list(folder.rglob("*.shp"))[
            0
        ]  # todo: refactor as geojsons can be too
        # logging.warning("doing recursive .shp file search, maybe need to disable")
    except:
        raise ValueError(f"no shp file found in folder: {folder.name}")
    print(f"geom file: {geom_file}")
    if ref_raster_filename is None:
        # todo: refactor
        try:
            ref_raster_file = raster_folder_path / f"{folder.name}.tif"
            if not ref_raster_file.exists():
                print("here", raster_folder_path)
                try:
                    ref_raster_file = find_ref_file(raster_folder_path)[0]
                except:
                    ref_raster_file = find_ref_file(
                        Path(str(raster_folder_path)[:-3])
                    )[0]
        except:
            ref_raster_file = (
                raster_folder_path.parent
                / f"{folder.name[:-3]}"
                / f"{folder.name[:-3]}.tif"
            )
            if not ref_raster_file.exists():
                ref_raster_file = (
                    raster_folder_path.parent
                    / f"{folder.name[:-3]}"
                    / f"{folder.name[:-3]}.jp2"
                )
    else:
        ref_raster_file = raster_folder_path / ref_raster_filename

    if dst_folder_path is None:
        new_filename = ref_raster_file.parent / dst_file_name
    else:
        new_filename = dst_folder_path / dst_file_name

    rasterize_file(geom_file, ref_raster_file, new_filename)


@validate_arguments
def rasterize(
    geom_folder_path: Path,
    raster_folder_path: Path,
    ref_raster_filename=None,
    dst_folder_path: Optional[Path] = None,
    dst_filename: str = "label.tif",
):
    # find files/folders in the folder
    folders = geom_folder_path.iterdir()
    for folder in tqdm(folders):
        rasterize_single(
            folder,
            raster_folder_path / folder.name,
            ref_raster_filename,  # ref_raster_filename /  / folder.name  #
            dst_folder_path
            if dst_folder_path is None
            else dst_folder_path / folder.name,
            dst_filename,
        )


if __name__ == "__main__":
    Fire(rasterize)
