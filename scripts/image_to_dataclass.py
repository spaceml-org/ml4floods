# ====================================================
# Helpful trick for loading the directories correction
# ====================================================
from pathlib import Path
import sys, os
from pyprojroot import here

# spyder up to find the root
root = here(project_files=[".here"])
# append to path
sys.path.append(str(here()))

from dataclasses import dataclass, field
from typing import Dict, Tuple, List
from datetime import datetime
import rasterio

# ====================================================
# Utils for getting filenames and directories
# ====================================================
from src.data.utils import (
    get_files_in_directory,
    get_filenames_in_directory,
    get_files_in_bucket_directory,
)
from src.data.worldfloods.dataclass_.baseclass import WorldFloodsS2ImageSaved
from src.data.worldfloods.dataclass_.hardutils import (
    parse_gcp_files_dataclass,
    open_source_tiff_meta,
    open_source_meta,
    open_source_tiff,
)
from src.data.utils import open_file_from_bucket
import tqdm
from src.data.worldfloods.dataclass_.utils import (
    load_dataclass_pickle,
    save_dataclass_pickle,
)
from src.data.utils import create_folder
import pickle
from src.data.utils import save_file_to_bucket

# ====================================================
# Standard Packages
# ====================================================
# from torchvision import transforms
import numpy as np


from src.data.utils import get_files_in_bucket_directory

# define initial parameters
bucket_id = "ml4floods"
directory = "worldfloods/tiffimages/S2"
suffix = ".tif"

# extract all files in directory
files = get_files_in_bucket_directory(bucket_id, directory, suffix)

# hack to make the files include the index
files = [f"gs://{bucket_id}/{ifile}" for ifile in files]

print(f"Number of files: {len(files)}")
print(f"Demo filename:\n'{files[0]}'")


def init_wfs2_dataclass(full_path: str) -> dataclass:
    """Function to initialize and return the WorldFloodsS2ImageSaved dataclass.

    Args:
        full_path (str): Path of the tiff file.

    Returns:
        dataclass: Returns dataclass object of the mentioned tiff file.
    """
    # parse the components

    # initialize dataclass
    dc = WorldFloodsS2ImageSaved(full_path=full_path)

    dc = parse_gcp_files_dataclass(dc)

    dc = open_source_tiff_meta(dc)
    try:
        dc = open_source_meta(dc)
    except AttributeError:
        print(f"Metadata for {dc.file_name} not found")

    dc = open_source_tiff(dc)

    return dc


for ifile in tqdm.tqdm(files):

    # initialize
    dc_example = init_wfs2_dataclass(ifile)

    # save name
    save_name = Path(dc_example.file_name).stem
    suffix = ".pkl"

    # local directory save
    local_dir = root.joinpath("datasets/test/")
    local_dir = str(local_dir.joinpath(save_name + suffix))
    save_dataclass_pickle(dc_example, local_dir)

    # bucket save
    target_dir = f"gs://ml4cc_data_lake/0_DEV/0_Raw/WorldFloods/tiffimages_dataclass/{save_name}{suffix}"
    save_file_to_bucket(target_dir, local_dir)

    # remove from local directory
    rem_file = Path(local_dir)
    rem_file.unlink()

    del dc_example