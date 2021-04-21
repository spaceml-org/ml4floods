import tqdm
from ml4floods.data.utils import GCPPath, write_json_to_gcp
from pathlib import Path
from ml4floods.data.worldfloods.create_worldfloods_dataset import generate_item, worldfloods_output_files
from ml4floods.data.create_gt import generate_land_water_cloud_gt, generate_water_cloud_binary_gt


def main(version="v1_0",overwrite=False, prod_dev="0_DEV"):

    assert version in ["v1_0", "v2_0"], f"Unexpected version {version}"
    assert prod_dev in ["0_DEV", "2_PROD"], f"Unexpected environment {prod_dev}"

    ml_paths = [
        "test",
        "val",
        "train"
    ]
    destination_bucket_id = "ml4cc_data_lake"
    destination_parent_path = f"{prod_dev}/2_Mart/worldfloods_{version}"
    if version.startswith("v1"):
        gt_fun = generate_land_water_cloud_gt
    elif version.startswith("v2"):
        gt_fun = generate_water_cloud_binary_gt
    else:
        raise NotImplementedError(f"version {version} not implemented")

    problem_files = []

    dict_splits = {}
    for ipath in ml_paths:
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
                path_write = Path(destination_bucket_id).joinpath(destination_parent_path).joinpath(ipath)
                status = generate_item(s2_image_path, path_write,
                                       overwrite=overwrite,
                                       pbar=pbar, gt_fun=gt_fun)
                if status:
                    s2path = GCPPath(s2_image_path)
                    cloudprob_path_dest, floodmap_path_dest, gt_path_dest, \
                    meta_parent_path, permanent_water_image_path_dest, s2_image_path_dest = worldfloods_output_files(
                        output_path=path_write, tiff_file_name=s2path.file_name, permanent_water_available=True)
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
                        help="Which version of the data we want to create (3-class) or multioutput binary")
    parser.add_argument('--prod_dev', default='v1_0', choices=["0_DEV", "2_PROD"],
                        help="environment where the dataset would be created")
    parser.add_argument('--overwrite', default=False, action='store_true',
                        help="Overwrite the content in the folder {prod_dev}/2_Mart/worldfloods_{version}")

    args = parser.parse_args()

    main(version=args.version, overwrite=args.overwrite, prod_dev=args.prod_dev)