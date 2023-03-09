# third party libraries
import pytest
import rasterio as rio

from innofw.utils.data_utils.preprocessing.band_composer import BandComposer
from innofw.utils.data_utils.preprocessing.band_composer import (
    Landsat8BandComposer,
)
from innofw.utils.data_utils.preprocessing.band_composer import (
    Sentinel2BandComposer,
)
from tests.utils import get_test_folder_path

# local modules


@pytest.fixture
def root_path():
    return (
        get_test_folder_path()
        / "data"
        / "images"
        / "other"
        / "satellite_cropped"
    )


@pytest.fixture
def channels():
    return ["RED", "GRN", "BLU"]


@pytest.mark.parametrize(
    ["src_path", "dst_path", "band_composer", "ref_file"],
    [
        [
            "sentinel2/one",
            "s2.tif",
            Sentinel2BandComposer(),
            "GRANULE/L1C_T39UXB_A026401_20200712T074155/IMG_DATA/T39UXB_20200712T073621_B04.jp2",
        ],
        ["prepared/one", "prep.tif", BandComposer(), "BLU.jp2"],
        [
            "landsat8/one",
            "l8.tif",
            Landsat8BandComposer(),
            "LC08_L1TP_165017_20210812_20210819_01_T1_B4.TIF",
        ],
    ],
)
def test_sentinel2_band_composer(
    src_path, dst_path, band_composer, ref_file, channels, root_path, tmp_path
):
    src_path = root_path / src_path
    dst_path = tmp_path / dst_path

    assert not dst_path.exists()

    band_composer.compose_bands(src_path, dst_path, channels)

    assert dst_path.exists()

    raster = rio.open(dst_path)
    assert raster.meta["count"] == len(channels)

    band = src_path / ref_file
    band_ds = rio.open(band)
    assert raster.meta["width"] == band_ds.meta["width"]
    assert raster.meta["height"] == band_ds.meta["height"]
