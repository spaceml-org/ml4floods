import tqdm
from ml4floods.data.utils import GCPPath, write_json_to_gcp
from pathlib import Path
from ml4floods.data.worldfloods.create_worldfloods_dataset import generate_item, worldfloods_output_files, worldfloods_extra_gcp_paths
from ml4floods.data.create_gt import generate_land_water_cloud_gt, generate_water_cloud_binary_gt
from ml4floods.data import utils
import os
import fsspec
import json
import warnings


def main(version="v1_0",overwrite=False, prod_dev="0_DEV", dataset="original"):

    assert version in ["v1_0", "v2_0"], f"Unexpected version {version}"
    assert prod_dev in ["0_DEV", "2_PROD"], f"Unexpected environment {prod_dev}"
    assert dataset in ["original", "extra"], f"Unexpected dataset {dataset}"

    destination_bucket_id = "ml4cc_data_lake"
    if dataset == "original":
        destination_parent_path = f"{prod_dev}/2_Mart/worldfloods_{version}"
    else:
        destination_parent_path = f"{prod_dev}/2_Mart/worldfloods_{dataset}_{version}"
    if version.startswith("v1"):
        gt_fun = generate_land_water_cloud_gt
    elif version.startswith("v2"):
        gt_fun = generate_water_cloud_binary_gt
    else:
        raise NotImplementedError(f"version {version} not implemented")

    if dataset == "original":
        main_worldlfoods_original(destination_bucket_id, destination_parent_path, overwrite, gt_fun)

    if dataset == "extra":
        main_worldlfoods_extra(destination_bucket_id, destination_parent_path, overwrite,prod_dev, gt_fun)



def main_worldlfoods_extra(destination_bucket_id, destination_parent_path, overwrite, prod_dev, gt_fun):

    fs = fsspec.filesystem("gs")

    problem_files = []

    with fs.open(f"gs://ml4cc_data_lake/{prod_dev}/2_Mart/worldfloods_v1_0/train_test_split.json", "r") as fh:
        data = json.load(fh)

    train_val_test_split = {}
    for split in ["train", "test", "val"]:
        train_val_test_split[split] = set((os.path.splitext(os.path.basename(d))[0] for d in data[split]["S2"]))

    cems_codes_test = set(s.split("_")[0] for s in train_val_test_split["test"])
    cems_codes_test.add("EMSR9284")
    cems_codes_test.add("EMSR284")

    # get all files
    files_metadata_pickled = [f"gs://{f}" for f in fs.glob(f"gs://ml4cc_data_lake/{prod_dev}/1_Staging/WorldFloods/*/*/flood_meta/*.pickle")]

    # loop through files in the bucket
    with tqdm.tqdm(files_metadata_pickled, desc="Generating ground truth extra data") as pbar:
        for metadata_file in pbar:
            metadata_floodmap = utils.read_pickle_from_gcp(metadata_file)
            event_id = metadata_floodmap["event id"]+"_observed_event_a" # add observed_event_a for backwards compatibility

            # Find out which split to put the data in
            subset = "unused"
            for split in ["train", "test", "val"]:
                if event_id in train_val_test_split[split]:
                    subset = split
                    if split == "test":
                        expected_test_file = f"gs://ml4cc_data_lake/2_Mart/worldfloods_v1_0/test/floodmaps/{event_id}.geojson"
                        if fs.exists(expected_test_file):
                            metadata_floodmap["floodmap"] = expected_test_file
                        else:
                            warnings.warn(f"Test file {event_id} does not exists in old test database {expected_test_file}")
                    break
                if (split == "test") and metadata_floodmap["ems_code"] in cems_codes_test:
                    subset = "banned"

            path_write = os.path.join(destination_bucket_id, destination_parent_path, subset)
            status = generate_item(metadata_file,
                                   path_write,
                                   file_name=event_id,
                                   overwrite=overwrite,
                                   pbar=pbar, gt_fun=gt_fun,
                                   paths_function=worldfloods_extra_gcp_paths)
            if not status:
                problem_files.append(path_write)


    print("Files not generated that were expected:")
    for p in problem_files:
        print(p)


def main_worldlfoods_original(destination_bucket_id, destination_parent_path, overwrite,
                              gt_fun):
    problem_files = []

    dict_splits = {}
    for ipath in ["test", "val","train"]:
        dict_splits[ipath] = {"S2":[], "gt": []}

        # ensure path name is the same as ipath for the loop
        demo_image = "gs://ml4floods/worldfloods/public/test/S2/EMSR286_08ITUANGONORTH_DEL_MONIT02_v1_observed_event_a.tif"
        demo_image_gcp = GCPPath(demo_image)
        demo_image_gcp = demo_image_gcp.replace("test", ipath)

        # get all files in the parent directory
        files_in_bucket = demo_image_gcp.get_files_in_parent_directory_with_suffix(".tif")

        # loop through files in the bucket
        print(f"Generating ML GT for {ipath}, {len(files_in_bucket)} files")
        with tqdm.tqdm(files_in_bucket) as pbar:
            for s2_image_path in pbar:
                path_write = os.path.join(destination_bucket_id, destination_parent_path, ipath)
                status = generate_item(s2_image_path, path_write,
                                       file_name=os.path.splitext(os.path.basename(s2_image_path))[0],
                                       overwrite=overwrite,
                                       pbar=pbar, gt_fun=gt_fun)
                if status:
                    s2path = GCPPath(s2_image_path)
                    cloudprob_path_dest, floodmap_path_dest, gt_path_dest, \
                    meta_parent_path, permanent_water_image_path_dest, s2_image_path_dest = worldfloods_output_files(
                        output_path=path_write, file_name=s2path.file_name, permanent_water_available=True)
                    dict_splits[ipath]["S2"].append(s2_image_path_dest.full_path)
                    dict_splits[ipath]["gt"].append(gt_path_dest.full_path)
                else:
                    problem_files.append(path_write)

    # Save train_test split file
    path_splits = GCPPath(str(Path(destination_bucket_id).joinpath(destination_parent_path).joinpath("train_test_split.json")))
    write_json_to_gcp(path_splits.full_path, dict_splits)

    print("Files not generated that were expected:")
    for p in problem_files:
        print(p)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser('Generate WorldFloods ML Dataset')
    parser.add_argument('--version', default='v1_0', choices=["v1_0", "v2_0"],
                        help="Which version of the ground truth we want to create (3-class) or multioutput binary")
    parser.add_argument('--dataset', default='original', choices=["original", "extra"],
                        help="Use the original data or the newly downloaded data from Copernicus EMS")
    parser.add_argument('--prod_dev', default='0_DEV', choices=["0_DEV", "2_PROD"],
                        help="environment where the dataset would be created")
    parser.add_argument('--overwrite', default=False, action='store_true',
                        help="Overwrite the content in the folder {prod_dev}/2_Mart/worldfloods_{version}")

    args = parser.parse_args()

    main(version=args.version, overwrite=args.overwrite, prod_dev=args.prod_dev, dataset=args.dataset)
