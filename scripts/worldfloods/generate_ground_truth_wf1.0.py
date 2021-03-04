import sys, os
from pyprojroot import here

# spyder up to find the root
root = here(project_files=[".here"])

# append to path
sys.path.append(str(here()))

import logging
import json
from src.data.create_gt import (
    _get_image_geocoords,
    generate_land_water_cloud_gt,
    generate_water_cloud_binary_gt,
)
from src.data.io import save_groundtruth_tiff_rasterio
import os
from src.data.utils import GCPPath
from src.data.config import BANDS_S2, CODES_FLOODMAP, UNOSAT_CLASS_TO_TXT

import rasterio.windows
from pathlib import Path

import tqdm
from src.data.utils import save_file_to_bucket


def main():

    # looping through the ML parts
    ml_paths = [
        # "test",
        "val",
        "train",
    ]

    local_path = Path(root).joinpath("datasets")

    bucket_id = "ml4floods"
    destination_bucket_id = "ml4cc_data_lake"

    parent_path = "worldfloods/public"
    destination_parent_path = "0_DEV/2_Mart/worldfloods_v1_0"

    demo_image = "gs://ml4floods/worldfloods/public/test/S2/EMSR286_08ITUANGONORTH_DEL_MONIT02_v1_observed_event_a.tif"

    for ipath in ml_paths:

        # want the appropate ml path
        demo_image_gcp = GCPPath(demo_image)

        # ensure path name is the same as ipath for the loooop
        demo_image_gcp = demo_image_gcp.replace("test", ipath)

        # get all files in the parent directory
        files_in_bucket = demo_image_gcp.get_files_in_parent_directory_with_suffix(
            ".tif"
        )

        # loop through files in the bucket
        print(f"Generating ML GT for {ipath.title()}")
        with tqdm.tqdm(files_in_bucket) as pbar:
            for s2_image_path in pbar:

                s2_image_path = GCPPath(s2_image_path)

                # create floodmap path
                floodmap_path = s2_image_path.replace("/S2/", "/floodmaps/")
                floodmap_path = floodmap_path.replace(".tif", ".shp")

                # create meta path
                meta_path = s2_image_path.replace("/S2/", "/meta/")
                meta_path = meta_path.replace(".tif", ".json")

                # ==============================
                # Generate GT Image
                # ==============================
                pbar.set_description("Generating Ground Truth...")
                # generate gt and gt meta
                gt, gt_meta = generate_land_water_cloud_gt(
                    s2_image_path.full_path, floodmap_path.full_path, keep_streams=True
                )

                # ==============================
                # SAVE S2 Image
                # ==============================
                pbar.set_description("Saving S2 image...")

                # download file locally
                local_file_name = s2_image_path.download_file_from_bucket(
                    str(local_path)
                )
                # change file name
                demo_s2_image_bucket = s2_image_path.replace(
                    bucket_id, destination_bucket_id
                )
                demo_s2_image_bucket = demo_s2_image_bucket.replace(
                    parent_path, destination_parent_path
                )

                # save file to bucket
                save_file_to_bucket(demo_s2_image_bucket.full_path, local_file_name)

                # delete the local file
                Path(local_file_name).unlink()

                # # get parent path name
                # s2_image_parent_destination = (
                #     Path(destination_parent_path).joinpath(ipath).joinpath("S2")
                # )
                # s2_image_path.transfer_file_to_bucket(
                #     destination_bucket_id, s2_image_parent_destination
                # )

                # save_file_to_bucket(
                #     gt_path.full_path, str(local_path.joinpath(gt_path.file_name))
                # )

                # ==============================
                # SAVE Meta Data
                # ==============================
                pbar.set_description("Saving meta data...")
                # get parent path name
                meta_parent_destination = (
                    Path(destination_parent_path).joinpath(ipath).joinpath("meta")
                )
                meta_path.transfer_file_to_bucket(
                    destination_bucket_id, meta_parent_destination
                )

                # ==============================
                # SAVE FloodMap Data
                # ==============================
                # special case of multiple files
                pbar.set_description("Saving floodmap meta data...")

                # get parent path name
                floodmap_parent_destination = (
                    Path(destination_parent_path).joinpath(ipath).joinpath("floodmap")
                )

                floodmap_meta_files = (
                    floodmap_path.get_files_in_parent_directory_with_name()
                )

                for ifloodmap_meta_file in floodmap_meta_files:
                    GCPPath(ifloodmap_meta_file).transfer_file_to_bucket(
                        destination_bucket_id, floodmap_parent_destination
                    )

                # ==============================
                # SAVE GT Data (WorldFloods 1.1)
                # ==============================
                pbar.set_description("Saving GT data...")

                # replace parent path
                gt_path = demo_image_gcp.replace(bucket_id, destination_bucket_id)
                gt_path = gt_path.replace("/S2/", "/gt/")
                gt_path = gt_path.replace(parent_path, destination_parent_path)

                # save ground truth
                save_groundtruth_tiff_rasterio(
                    gt,
                    str(local_path.joinpath(gt_path.file_name)),
                    gt_meta=gt_meta,
                    crs=gt_meta["crs"],
                    transform=gt_meta["transform"],
                )
                save_file_to_bucket(
                    gt_path.full_path, str(local_path.joinpath(gt_path.file_name))
                )
                # delate local file
                local_path.joinpath(gt_path.file_name).unlink()


if __name__ == "__main__":
    main()