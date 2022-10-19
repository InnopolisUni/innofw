# standard libraries
import os
from abc import ABC, abstractmethod
from typing import Dict
from math import radians
from pathlib import Path
from xml.etree import ElementTree

# third party libraries
from pydantic import DirectoryPath, FilePath


class BaseSatelliteSource(ABC):
    @property
    def metadata(self):
        file = self.find_metadata_file()
        _metadata = self.parse_metadata_file(file)
        return _metadata

    @abstractmethod
    def find_metadata_file(self) -> Path:
        pass

    @abstractmethod
    def parse_metadata_file(self, metadata_file: FilePath) -> dict:
        pass


class Sentinel2(BaseSatelliteSource):
    def __init__(self, src_folder: DirectoryPath):
        self.src_folder = src_folder

    def find_metadata_file(self) -> Path:
        files = list(self.src_folder.rglob("MTD_MSIL1C.xml"))

        if len(files) < 1:
            raise ValueError(
                f"Unable to find metadata file in the landsat 8 folder: {self.src_folder}"
            )
        return files[0]

    def parse_metadata_file(self, metadata_file: FilePath) -> dict:
        tree = ElementTree.parse(metadata_file)
        bands = self._get_bands_from_tree(tree)
        assert len(bands) > 0
        bands = {key: self.src_folder / band_path for key, band_path in bands.items()}
        metadata = {
            "bands": bands,
            "mapping": self._construct_band_mapping(bands),
            "num_bands": len(bands),
            "date_acquired": self._get_date_acquired_from_tree(tree),
        }
        return metadata

    def _get_bands_from_tree(self, tree) -> Dict[int, str]:
        used_bands = [
            node.text
            for node in tree.findall(".//IMAGE_FILE")
            if node.text[-3:] not in self.UNUSED_BANDS
        ]
        indexed_bands = {
            band_index: f"{band_name}.tif"
            for band_index, band_name in enumerate(used_bands, start=1)
        }
        for band_index, band_name in indexed_bands.items():
            band_path = self.src_folder / band_name
            if not band_path.exists():
                indexed_bands[band_index] = band_name[:-3] + "jp2"
        return indexed_bands

    @staticmethod
    def _construct_band_mapping(bands: Dict[int, str]) -> Dict[str, int]:
        band_mapping = {
            os.path.splitext(name)[0][-3:]: band_index
            for band_index, name in bands.items()
        }
        return band_mapping

    def _get_date_acquired_from_tree(self, tree) -> str:
        # Sentinel 2 does not seem to use the term "acquisition date" but
        # "generation date" appears to have the closest meaning.
        return tree.find(".//GENERATION_TIME").text[:10]

    UNUSED_BANDS = {"TCI"}


class Landsat8(BaseSatelliteSource):
    def __init__(self, src_folder: DirectoryPath):
        self.src_folder = src_folder

    def find_metadata_file(self) -> Path:
        files = list(self.src_folder.rglob("*MTL.TXT")) + list(
            self.src_folder.rglob("*MTL.txt")
        )

        if len(files) < 1:
            raise ValueError(
                f"Unable to find metadata file in the landsat 8 folder: {self.src_folder}"
            )
        return files[0]

    def parse_metadata_file(self, metadata_file: FilePath) -> dict:
        bands: Dict[int, str] = dict()
        band_mapping: Dict[str, int] = dict()
        mul_factors: Dict[int, float] = dict()
        add_factors: Dict[int, float] = dict()
        sun_elevation = 0.0
        date_acquired = "1970-01-01"

        with open(metadata_file) as file:
            for line in file:
                try:
                    key, value = line.strip().split(" = ")
                except ValueError:
                    continue

                if (
                    key.startswith("FILE_NAME_BAND_")
                    and key != "FILE_NAME_BAND_QUALITY"
                ):
                    band_index = self._band_index_from_key(key)
                    bands[band_index] = self._remove_quotes(value)
                    band_tag = self._band_tag_from_index(band_index)
                    band_mapping[band_tag] = band_index
                elif key.startswith("REFLECTANCE_MULT_BAND_"):
                    band_index = self._band_index_from_key(key)
                    mul_factors[band_index] = float(value)
                elif key.startswith("REFLECTANCE_ADD_BAND_"):
                    band_index = self._band_index_from_key(key)
                    add_factors[band_index] = float(value)
                elif key == "SUN_ELEVATION":
                    sun_elevation = radians(float(value))
                elif key == "DATE_ACQUIRED":
                    date_acquired = value

        assert len(bands) > 0
        bands = {key: self.src_folder / band_path for key, band_path in bands.items()}
        _metadata = {
            "bands": bands,
            "mapping": band_mapping,
            "num_bands": len(bands),
            "mul_factors": mul_factors,
            "add_factors": add_factors,
            "sun_elevation": sun_elevation,
            "date_acquired": date_acquired,
        }
        return _metadata

    @staticmethod
    def _band_index_from_key(key: str) -> int:
        return int(key.split("_")[-1])

    @staticmethod
    def _remove_quotes(value: str) -> str:
        return value[1:-1]

    @staticmethod
    def _band_tag_from_index(index: int) -> str:
        return f"B{index:02}"
