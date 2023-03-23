#
import shutil
from pathlib import Path

from tqdm import tqdm

a = Path(
    "/mnt/nvmestorage/qb/data_n_weights/linear-road-bin-seg-oftp/301122/processed/070123-167folders-2048"
)
b = Path(
    "/mnt/nvmestorage/qb/data_n_weights/linear-road-bin-seg-oftp/200323/processed/2023-03-21-2048/train"
)
dst_path = Path(
    "/mnt/nvmestorage/qb/data_n_weights/linear-road-bin-seg-oftp/200323/processed/2023-03-21-2048-merged-with-old/train"
)


def copy_files(folder, dst_path):
    for i in tqdm(folder.glob("*.tif")):
        if (dst_path / i.name).exists():
            new_file_path = dst_path / f"{i.stem}_1{i.suffix}"
        else:
            new_file_path = dst_path / i.name
        shutil.copyfile(i, new_file_path)


copy_files(a / "images", dst_path / "images")
copy_files(a / "masks", dst_path / "masks")
copy_files(b / "images", dst_path / "images")
copy_files(b / "masks", dst_path / "masks")
