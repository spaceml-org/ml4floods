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
from src.data.utils import GCPPath, load_json_from_bucket
from src.data.config import BANDS_S2, CODES_FLOODMAP, UNOSAT_CLASS_TO_TXT

import rasterio.windows
from pathlib import Path

import tqdm
from src.data.utils import save_file_to_bucket


def main():

    # looping through the ML parts
    ml_paths = [
        # "val",
        # "test",
        "train",
    ]

    local_path = Path(root).joinpath("datasets")

    bucket_id = "ml4floods"
    destination_bucket_id = "ml4cc_data_lake"

    parent_path = "worldfloods/public"
    destination_parent_path = "0_DEV/2_Mart/worldfloods_v2_0"
    cloud_prob_parent_path = "worldfloods/tiffimages"
    permanent_water_parent_path = "worldfloods/tiffimages/PERMANENTWATERJRC"
    save_s2_image = False
    save_meta_data = False
    save_permanent_water_image = False
    save_cloud_prob = False
    save_floodmap_meta = False

    demo_image = "gs://ml4floods/worldfloods/public/test/S2/EMSR286_08ITUANGONORTH_DEL_MONIT02_v1_observed_event_a.tif"

    # want the appropate ml path

    problem_files = []

    for ipath in ml_paths:

        # ensure path name is the same as ipath for the loooop
        demo_image_gcp = GCPPath(demo_image)
        demo_image_gcp = demo_image_gcp.replace("test", ipath)

        # get all files in the parent directory
        files_in_bucket = demo_image_gcp.get_files_in_parent_directory_with_suffix(
            ".tif"
        )
        # # TESTINGGGGGGG
        # files_in_bucket = [
        #     "gs://ml4cc_data_lake/0_DEV/2_Mart/worldfloods_v2_0/train/S2/EMSR286_08ITUANGONORTH_DEL_MONIT02_v1_observed_event_a.tif"
        # ]
        # # HACK FOR SLICING
        last_x_slices = slice(-50, None)

        # loop through files in the bucket
        print(f"Generating ML GT for {ipath.title()}")

        with tqdm.tqdm(list(reversed(files_in_bucket[last_x_slices]))) as pbar:
            for s2_image_path in pbar:

                try:

                    pbar.set_description("Getting Paths...")

                    s2_image_path = GCPPath(s2_image_path)

                    # create floodmap path
                    floodmap_path = s2_image_path.replace("/S2/", "/floodmaps/")
                    floodmap_path = floodmap_path.replace(".tif", ".shp")

                    # create cloudprob path
                    try:

                        cloudprob_path = GCPPath(
                            str(
                                Path(bucket_id)
                                .joinpath(cloud_prob_parent_path)
                                .joinpath("cloudprob_edited")
                                .joinpath(s2_image_path.file_name)
                            )
                        )
                        assert cloudprob_path.check_if_file_exists() is True
                    except AssertionError:
                        cloudprob_path = GCPPath(
                            str(
                                Path(bucket_id)
                                .joinpath(cloud_prob_parent_path)
                                .joinpath("cloudprob")
                                .joinpath(s2_image_path.file_name)
                            )
                        )

                    # create meta path
                    meta_path = s2_image_path.replace("/S2/", "/meta/")
                    meta_path = meta_path.replace(".tif", ".json")

                    # ==============================
                    # OPEN PERMANENT WATER TIFF
                    # ==============================
                    try:
                        pbar.set_description("Grabbing Permanent Water Tiff...")
                        permenant_water_path = GCPPath(
                            str(
                                Path(bucket_id)
                                .joinpath(permanent_water_parent_path)
                                .joinpath(s2_image_path.file_name)
                            )
                        )
                        assert permenant_water_path.check_if_file_exists() is True
                        permenant_water_path = permenant_water_path.full_path

                    except AssertionError:
                        pbar.set_description("Didnt Find...")
                        permenant_water_path = None

                    # ==============================
                    # Generate GT Image
                    # ==============================
                    pbar.set_description("Generating Ground Truth...")

                    # load the meta
                    floodmap_meta = load_json_from_bucket(
                        meta_path.bucket_id, meta_path.get_file_path()
                    )

                    # generate gt and gt meta
                    # Run it through the GT script
                    gt, gt_meta = generate_water_cloud_binary_gt(
                        s2_image_path.full_path,
                        floodmap_path.full_path,
                        floodmap_meta,
                        keep_streams=True,
                        cloudprob_image_path=cloudprob_path.full_path,
                        permanent_water_image_path=permenant_water_path,
                    )
                    # ==============================
                    # SAVE Permanent Water Image
                    # ==============================

                    if save_permanent_water_image and permenant_water_path is not None:
                        pbar.set_description("Saving permanent water image...")

                        permenant_water_path = GCPPath(
                            str(
                                Path(bucket_id)
                                .joinpath(permanent_water_parent_path)
                                .joinpath(s2_image_path.file_name)
                            )
                        )
                        # NEW WAY!!!
                        permanent_water_image_path_dest = GCPPath(
                            str(
                                Path(destination_bucket_id)
                                .joinpath(destination_parent_path)
                                .joinpath(ipath)
                                .joinpath("permanent_water")
                                .joinpath(s2_image_path.file_name)
                            )
                        )

                        permenant_water_path.transfer_file_to_bucket_gsutils(
                            permanent_water_image_path_dest.full_path, file_name=True
                        )
                    # ==============================
                    # SAVE S2 Image
                    # ==============================
                    if save_s2_image:
                        pbar.set_description("Saving S2 image...")

                        # NEW WAY!!!
                        s2_image_path_dest = GCPPath(
                            str(
                                Path(destination_bucket_id)
                                .joinpath(destination_parent_path)
                                .joinpath(ipath)
                                .joinpath("S2")
                                .joinpath(s2_image_path.file_name)
                            )
                        )

                        s2_image_path.transfer_file_to_bucket_gsutils(
                            s2_image_path_dest.full_path, file_name=True
                        )

                    # ==============================
                    # SAVE Meta Data
                    # ==============================
                    if save_meta_data:
                        pbar.set_description("Saving meta data...")
                        # get parent path name
                        meta_parent_destination = (
                            Path(destination_parent_path)
                            .joinpath(ipath)
                            .joinpath("meta")
                        )
                        meta_path.transfer_file_to_bucket(
                            destination_bucket_id, meta_parent_destination
                        )
                    # ==============================
                    # SAVE Cloud Probabilities
                    # ==============================
                    if save_cloud_prob:
                        pbar.set_description("Saving cloud probs data...")
                        # get parent path name

                        cloudprob_path_dest = GCPPath(
                            str(
                                Path(destination_bucket_id)
                                .joinpath(destination_parent_path)
                                .joinpath(ipath)
                                .joinpath("cloudprob")
                                .joinpath(cloudprob_path.file_name)
                            )
                        )

                        cloudprob_path.transfer_file_to_bucket_gsutils(
                            cloudprob_path_dest.full_path, file_name=True
                        )
                    # ==============================
                    # SAVE FloodMap Data
                    # ==============================
                    if save_floodmap_meta:
                        # special case of multiple files
                        pbar.set_description("Saving floodmap meta data...")

                        # get parent path name
                        floodmap_parent_destination = (
                            Path(destination_parent_path)
                            .joinpath(ipath)
                            .joinpath("floodmap")
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
                    # print("here!")
                    pbar.set_description("Saving GT data...")

                    # replace parent path
                    gt_path = s2_image_path.replace(bucket_id, destination_bucket_id)
                    gt_path = gt_path.replace("/S2/", "/gt/")
                    gt_path = gt_path.replace(parent_path, destination_parent_path)
                    ##################################
                    # PLOTTING (FOR DEBUGGING)
                    ##################################
                    # import matplotlib.pyplot as plt
                    # from rasterio import plot as rasterioplt

                    # fig, ax = plt.subplots()
                    # rasterioplt.show(gt[1], transform=gt_meta["transform"], ax=ax)
                    # fig.savefig("./temp_water.png")

                    # save ground truth
                    save_groundtruth_tiff_rasterio(
                        gt,
                        str(local_path.joinpath(gt_path.file_name)),
                        gt_meta=None,
                        crs=gt_meta["crs"],
                        transform=gt_meta["transform"],
                    )
                    save_file_to_bucket(
                        gt_path.full_path, str(local_path.joinpath(gt_path.file_name))
                    )
                    # delate local file
                    local_path.joinpath(gt_path.file_name).unlink()

                except KeyboardInterrupt:
                    break
                except:
                    problem_files.append(s2_image_path.full_path)

    print(problem_files)

    import pickle

    with open("./momoney_moprobs_v2.pickle", "wb") as fp:
        pickle.dump(problem_files, fp)


if __name__ == "__main__":
    main()