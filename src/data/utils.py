"""
Demo script to download some demo data files. Mainly used for testing but can also be used for other explorations.
"""
import argparse
import subprocess
from pathlib import Path

import rasterio
from google.cloud import storage

HOME = str(Path.home())


def save_file_from_bucket(bucket_id: str, file_name: str, destination_file_path: str):
    """Saves a file from a bucket
    
    Parameters
    ----------
    bucket_id : str
        the name of the bucket
    file_name : str
        the name of the file in bucket (include the directory)
    destination_file_path : str
        the directory of where you want to save the
        data locally (not including the filename)
    
    Examples
    --------
    
    >>> bucket_id = ...
    >>> file_name = 'path/to/file/and/file.csv'
    >>> dest = 'path/in/bucket/'
    >>> load_file_from_bucket(
        bucket_id=bucket_id,
        file_name=file_name,
        destimation_file_path=dest
    )
    """
    client = storage.Client()

    bucket = client.get_bucket(bucket_id)
    # get blob
    blob = bucket.get_blob(file_name)

    # create directory if needed
    create_folder(destination_file_path)

    # get full path
    destination_file_name = Path(destination_file_path).joinpath(
        file_name.split("/")[-1]
    )
    # download data
    blob.download_to_filename(str(destination_file_name))

    return None


def check_path_exists(path: str) -> None:
    if not Path(path).is_dir():
        raise ValueError(f"Unrecognized path: {str(Path(path))}")
    return None

def create_folder(directory: str) -> None:
    """Creates directory if doesn't exist
    
    params:
        directory (str): a directory to be created if
            it doesn't already exist.

        Typical usage example:

        >>> from .src.data.utils import create_folder
        >>> directory = "./temp"
        >>> create_folder(directory)
    """
    try:
        Path(directory).mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"Folder '{directory}' Is Already There.")
    else:
        print(f"Folder '{directory}' is created.")