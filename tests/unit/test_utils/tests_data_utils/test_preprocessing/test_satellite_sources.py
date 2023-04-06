import pytest
from unittest.mock import MagicMock
from unittest.mock import patch
from tempfile import TemporaryDirectory

from innofw.utils.data_utils.preprocessing.satellite_sources import *


class TestBaseSatelliteSource:
    @pytest.fixture
    def base_satellite_source(self):
        return BaseSatelliteSource()

    def test_find_metadata_file(self, base_satellite_source):
        with pytest.raises(NotImplementedError):
            base_satellite_source.find_metadata_file()

    def test_parse_metadata_file(self, base_satellite_source):
        with pytest.raises(NotImplementedError):
            base_satellite_source.parse_metadata_file('')


class TestSentinel2:
    @pytest.fixture
    def sentinel2(self):
        with TemporaryDirectory() as tmp:
            yield Sentinel2(tmp)

    def test_find_metadata_file(self, sentinel2):
        with pytest.raises(ValueError):
            sentinel2.find_metadata_file()

        with TemporaryDirectory() as tmp:
            metadata_path = Path(tmp) / 'MTD_MSIL1C.xml'
            metadata_path.touch()

            sentinel2.src_folder = Path(tmp)
            assert sentinel2.find_metadata_file() == metadata_path

    def test_parse_metadata_file(self, sentinel2):
        sentinel2.src_folder = Path('')
        sentinel2._get_bands_from_tree = MagicMock()
        sentinel2._construct_band_mapping = MagicMock()
        sentinel2._get_date_acquired_from_tree = MagicMock()

        with patch('xml.etree.ElementTree.parse') as parse_mock:
            parse_mock.return_value = MagicMock()
            assert sentinel2.parse_metadata_file(Path('')) == {
                'bands': sentinel2._get_bands_from_tree.return_value,
                'mapping': sentinel2._construct_band_mapping.return_value,
                'num_bands': len(sentinel2._get_bands_from_tree.return_value),
                'date_acquired': sentinel2._get_date_acquired_from_tree.return_value,
            }


class TestLandsat8:
    @pytest.fixture
    def landsat8(self):
        with TemporaryDirectory() as tmp:
            yield Landsat8(tmp)

    def test_find_metadata_file(self, landsat8):
        with pytest.raises(ValueError):
            landsat8.find_metadata_file()

        with TemporaryDirectory() as tmp:
            metadata_path = Path(tmp) / 'LC08_L1TP_200027_20190702_20190709_01_T1_MTL.txt'
            metadata_path.touch()

            landsat8.src_folder = Path(tmp)
            assert landsat8.find_metadata_file() == metadata_path

    def test_parse_metadata_file(self, landsat8):
        with TemporaryDirectory() as tmp:
            metadata_path = Path(tmp) / 'LC08_L1TP_200027_20190702_20190709_01_T1_MTL.txt'
            metadata_path.touch()

            landsat8.src_folder = Path(tmp)

            assert landsat8.parse_metadata_file(metadata_path) == {
                'bands': {},
                'mapping': {},
                'num_bands': 0,
                'date_acquired': '1970-01-01',
            }
