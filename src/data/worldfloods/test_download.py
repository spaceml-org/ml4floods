"""
Demo script to download some demo data files. Mainly used for testing but can also be used for other explorations.
"""
import argparse
from src.data.utils import create_folder
from dataclasses import dataclass, field
from datetime import datetime
import subprocess
from pathlib import Path
from typing import Optional

import rasterio
from google.cloud import storage

from pyprojroot import here

root = here(project_files=[".here"])

HOME = root
from src.data.worldfloods.download import download_image, download_worldfloods_data

@dataclass
class WorldFloodsImage:
    # ESSENTIAL METADATA
    filename: str
    uri: str = field(default=None)
    filepath: str = field(default=None)
    bucket_id: str = field(default=None)
    product_id: str = field(default=None)

    # BREADCRUMBS
    load_date: str = field(default=datetime.now())
    viewed_by: list = field(default_factory=list, compare=False, repr=False)
    source_system: str = field(default="Not Specified")



def test_data_download(ml_split: str = "train"):

    # STEP 1 - Create Demo Directory

    # Step 2 - Download List of demo files
    bucket_id = "ml4floods"
    directory = "worldfloods/public/"

    files = [
        "01042016_Holmes_Creek_at_Vernon_FL.tif",
        "05302016_San_Jacinto_River_at_Porter_TX.tif",
        "05102017_Black_River_near_Pocahontas_AR0000012544-0000000000.tif",
    ]

    download_worldfloods_data(
        directories=files,
        destination_dir=str(Path(HOME).joinpath("datasets")),
        bucket_id=bucket_id,
        ml_split=ml_split,
    )


def download_demo_image(dest_dir: Optional[str] = None):

    if dest_dir is None:
        dest_dir = Path(HOME).joinpath("datasets/demo_images")
        create_folder(dest_dir)

    # ============
    # DATAIMAGE
    # ============

    # filename
    file_name = (
        "ml4floods/worldfloods/public/train/S2/01042016_Holmes_Creek_at_Vernon_FL.tif"
    )

    # initialize the dataclass (specific to the worldfloods images)
    dc_image_example = WorldFloodsImage(filename=file_name)

    # download image from bucket
    destination_dir = dest_dir.joinpath("S2")
    download_image(dc_image_example, destination_dir)

    # ============
    # GROUND TRUTH IMAGE
    # ============

    # filename
    file_name = (
        "ml4floods/worldfloods/public/train/gt/01042016_Holmes_Creek_at_Vernon_FL.tif"
    )

    # initialize the dataclass (specific to the worldfloods images)
    dc_gt_example = WorldFloodsImage(filename=file_name)

    # download image from bucket
    destination_dir = dest_dir.joinpath("gt")
    download_image(dc_gt_example, destination_dir)

    return None


def download_demo_trainsplit_image(dest_dir: Optional[str] = None):

    if dest_dir is None:
        dest_dir = Path(HOME).joinpath("datasets/demo_images")
        create_folder(dest_dir)

    # ============
    # DATAIMAGE
    # ============

    splits = ["train", "val", "test"]
    image_names = [
        "01042016_Holmes_Creek_at_Vernon_FL.tif",
        "RS2_20161008_Water_Extent_Corail_Pestel.tif",
        "EMSR286_09ITUANGOSOUTH_DEL_MONIT02_v1_observed_event_a.tif",
    ]

    # filename
    for i_split, i_image in zip(splits, image_names):
        file_name = f"ml4floods/worldfloods/public/{i_split}/S2/{i_image}"

        # initialize the dataclass (specific to the worldfloods images)
        dc_image_example = WorldFloodsImage(filename=file_name)

        # download image from bucket
        destination_dir = dest_dir.joinpath(i_split).joinpath("S2")
        download_image(dc_image_example, destination_dir)

        # ============
        # GROUND TRUTH IMAGE
        # ============

        # filename
        file_name = f"ml4floods/worldfloods/public/{i_split}/gt/{i_image}"

        # initialize the dataclass (specific to the worldfloods images)
        dc_gt_example = WorldFloodsImage(filename=file_name)

        # download image from bucket
        destination_dir = dest_dir.joinpath(i_split).joinpath("gt")
        download_image(dc_gt_example, destination_dir)

    return None


if __name__ == "__main__":
    test_data_download()