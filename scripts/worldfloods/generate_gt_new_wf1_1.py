import sys, os
from pyprojroot import here

# spyder up to find the root
root = here(project_files=[".here"])

# append to path
sys.path.append(str(here()))


from datetime import timedelta
from datetime import datetime
import pandas as pd
import geopandas as gpd
from pathlib import Path
import ee
from src.data import ee_download
from src.data.copernicusEMS import activations
from src.data.create_gt import generate_land_water_cloud_gt
from src.data.io import save_groundtruth_tiff_rasterio
from src.data.utils import (
    remove_gcp_prefix,
    get_files_in_directory_gcp,
    read_pickle_from_gcp,
    save_file_to_bucket,
)
from src.data.utils import GCPPath
from typing import Tuple
from collections import namedtuple
import tqdm


ActivationFile = namedtuple(
    "ActivationFile", ["directory_aoi", "event_activation", "file_name", "core_name"]
)

# 1. Get Activation Codes, .csv file
CSV_FILE = "gs://ml4cc_data_lake/0_DEV/0_Raw/WorldFloods/copernicus_ems/copernicus_ems_codes/ems_activations_20150701_20210304.csv"
TABLE_ACTIVATIONS_EMS = pd.read_csv(CSV_FILE, encoding="latin1").set_index("Code")
ESMR_CODES = list(TABLE_ACTIVATIONS_EMS[TABLE_ACTIVATIONS_EMS["has_aoi"] == True].index)

# predefined source directories
BUCKET_NAME = "ml4cc_data_lake"
PARENT_DIR_S2 = "0_DEV/1_Staging/WorldFloods/S2"
PARENT_DIR_JRC = "0_DEV/1_Staging/WorldFloods/JRC"
PARENT_DIR_FLOODS = "0_DEV/1_Staging/WorldFloods/floodmap"
PARENT_DIR_FLOOD_META = "0_DEV/1_Staging/WorldFloods/flood_meta"

# TODO: change this to be more robust
ALL_JRC_FILES = get_files_in_directory_gcp(BUCKET_NAME, PARENT_DIR_JRC)

# predefined target directories
LOCAL_PATH = Path(root).joinpath("datasets")
PARENT_DIR_GT = "0_DEV/1_Staging/WorldFloods/GT/V_1_1"


def find_leaf_nodes(filepath: str) -> Tuple[str, str]:

    filepath = remove_gcp_prefix(filepath)

    directory_aoi = str(Path(filepath).parents[0].parts[-1])
    event_aoi = str(Path(filepath).parents[1].parts[-1])
    file_name = str(Path(filepath).name)
    core_file_name = (
        "_".join(str(Path(file_name).name).split("_")[:5])
        + "_"
        + str(Path(file_name).name).split("_")[-1][:2]
    )
    return ActivationFile(
        directory_aoi=directory_aoi,
        event_activation=event_aoi,
        file_name=file_name,
        core_name=core_file_name,
    )


from time import sleep


def main():

    problem_files = []

    # 2. Loop Through Activation Codes
    with tqdm.tqdm(ALL_JRC_FILES) as pbar_files:

        for i_jrc_file in pbar_files:

            aoi_meta = find_leaf_nodes(i_jrc_file)

            # update progress bar
            pbar_files.set_description(
                f"ESMR Code: {aoi_meta.event_activation}, AOI: {aoi_meta.directory_aoi}"
            )

            # ======================
            # LOAD S2 IMAGE PATH
            # ======================
            pbar_files.set_description("Getting S2 Image Path...")
            s2_image_filepath = "gs://" + str(
                Path(BUCKET_NAME)
                .joinpath(PARENT_DIR_S2)
                .joinpath(aoi_meta.event_activation)
                .joinpath(aoi_meta.directory_aoi)
                .joinpath(aoi_meta.file_name)
            )

            if not GCPPath(s2_image_filepath).check_if_file_exists():

                # problem file
                problem_files.append("gs://" + i_jrc_file)
                continue

            # =======================
            # LOAD FLOODMAP, geojson
            # =======================

            # Load Floodmap geojson
            pbar_files.set_description("Getting Floodmap...")
            floodmap_geojson_path = "gs://" + str(
                Path(BUCKET_NAME)
                .joinpath(PARENT_DIR_FLOODS)
                .joinpath(aoi_meta.event_activation)
                .joinpath(aoi_meta.directory_aoi)
                .joinpath(aoi_meta.core_name + "_floodmap.geojson")
            )

            # ======================
            # LOAD FLOODMAP META
            # ======================
            pbar_files.set_description("Getting Floodmap meta...")
            meta_floodmap_filepath = "gs://" + str(
                Path(BUCKET_NAME)
                .joinpath(PARENT_DIR_FLOOD_META)
                .joinpath(aoi_meta.event_activation)
                .joinpath(aoi_meta.directory_aoi)
                .joinpath(aoi_meta.core_name + "_metadata_floodmap.pickle")
            )

            # ======================
            # LOAD GT
            # ======================
            pbar_files.set_description("Load Groundtruth...")
            gt, gt_meta = generate_land_water_cloud_gt(
                s2_image_filepath,
                floodmap_geojson_path,
                keep_streams=True,
                cloudprob_in_lastband=True,
                permanent_water_image_path="gs://" + i_jrc_file,
            )
            # ======================
            # SAVE GT (LOCALLY)
            # ======================
            pbar_files.set_description("Saving GT Locally...")
            LOCAL_DIR = LOCAL_PATH.joinpath(Path(s2_image_filepath).name)

            # save ground truth
            save_groundtruth_tiff_rasterio(
                gt,
                str(LOCAL_DIR),
                gt_meta=None,
                crs=gt_meta["crs"],
                transform=gt_meta["transform"],
            )
            # ======================
            # UPLOAD GT (GCP)
            # ======================
            pbar_files.set_description("Upload GT to bucket...")
            TARGET_DIR = "gs://" + str(
                Path(BUCKET_NAME)
                .joinpath(PARENT_DIR_GT)
                .joinpath(aoi_meta.event_activation)
                .joinpath(aoi_meta.directory_aoi)
                .joinpath(aoi_meta.file_name)
            )

            save_file_to_bucket(
                TARGET_DIR,
                str(LOCAL_DIR),
            )
            # delate local file
            LOCAL_DIR.unlink()

    import pickle

    with open("./problems_jrc.pickle", "wb") as fp:
        pickle.dump(problem_files, fp)


if __name__ == "__main__":
    main()
