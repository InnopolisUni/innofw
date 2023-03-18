# standard libraries
import hashlib

from pydantic import FilePath

# third party libraries


def compute_file_hash(file: FilePath, hash_type: str = "md5"):
    if hash_type == "md5":
        hash_func = hashlib.md5()
    elif hash_type == "sha256":
        hash_func = hashlib.sha256()
    else:
        raise ValueError("unresolved hash_type")

    chunk_size = 4096
    with open(file, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()
