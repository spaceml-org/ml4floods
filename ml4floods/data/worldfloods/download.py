import os
import sys
from pathlib import Path
from typing import List, Optional

from ml4floods.data.utils import download_data_from_bucket, save_file_from_bucket

BUCKET_ID = "ml4floods"
DIR = "worldfloods/public/"


def get_image_path(datclass) -> str:
    """Extracts the S2 Image path

    Args:
        ml_type (str): path to the dataset for the bucket
    Returns:
        path (str): path for the bucket
    """
    return str(Path(datclass.filename))


def download_image(datclass, destination: str, ml_type: str = "train") -> str:
    """Downloads the S2 Image

    Args:
        destination (str): path to where we want to save the data
        ml_type (str): path to the dataset for the bucket
    Returns:
        None
    """
    # get image path
    img_path = get_image_path(datclass)

    # download the image
    return download_data_from_bucket([img_path], destination)


def download_worldfloods_data(
    directories: List[str],
    destination_dir: str,
    ml_split: str = "train",
    bucket_id: Optional[str] = None,
) -> None:
    """Function to download data from the bucket to a local destination directory.
    This function differs from the save_file_from_bucket() function in that
    it takes as input a list of filenames to be downloaded compared to save_file_from_bucket()
    which deals with only a single file.
    Wraps around the save_file_from_bucket() function to download the list of files.

    Args:
        directories (List[str]): List of directories to be downloaded from the bucket.
        destination_dir (str): Path of the destination directory.
        ml_split (str, optional): The split that is to be downloaded.
            Defaults to "train".
            Options: train, val, test
        bucket_id (str, optional): Name of the source GCP bucket.
            Defaults to None.
    """

    if bucket_id is None:
        bucket_id = BUCKET_ID

    for ifile in directories:
        for iprefix in ["S2", "gt"]:

            # where to grab the file
            source = str(Path(DIR).joinpath(ml_split).joinpath(iprefix).joinpath(ifile))
            # Image where to save the file
            destination = Path(destination_dir).joinpath(ml_split).joinpath(iprefix)
            # copy file from bucket to savepath
            save_file_from_bucket(
                bucket_id=BUCKET_ID,
                file_name=source,
                destination_file_path=str(destination),
            )
