from pathlib import Path
from typing import List, Optional

from src.data.utils import save_file_from_bucket
import sys, os
from pyprojroot import here

# spyder up to find the root
root = here(project_files=[".here"])


HOME = root

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
