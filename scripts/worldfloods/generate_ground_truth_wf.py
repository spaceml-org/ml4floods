import json
import os
from collections.abc import Callable
from pathlib import Path

import fsspec
import pkg_resources
import tqdm

from ml4floods.data import utils
from ml4floods.data.create_gt import generate_land_water_cloud_gt, generate_water_cloud_binary_gt
from ml4floods.data.utils import write_json_to_gcp
from ml4floods.data.worldfloods.create_worldfloods_dataset import (
    generate_item,
    worldfloods_extra_gcp_paths,
    worldfloods_output_files,
)


def main(
    version="v1_0",
    overwrite=False,
    prod_dev="0_DEV",
    dataset="extra",
    cems_code="",
    aoi_code="",
    destination_parent_path="",
    subset="all",
):
    assert version in ["v1_0", "v2_0"], f"Unexpected ground truth version {version}"
    assert prod_dev in ["0_DEV", "2_PROD"], f"Unexpected environment {prod_dev}"
    assert dataset in ["original", "extra"], f"Unexpected dataset {dataset}"

    if not destination_parent_path:
        destination_parent_path = (
            f"gs://ml4cc_data_lake/{prod_dev}/2_Mart/worldfloods_{dataset}_{version}"
        )

    if version.startswith("v1"):
        gt_fun = generate_land_water_cloud_gt
    elif version.startswith("v2"):
        gt_fun = generate_water_cloud_binary_gt
    else:
        raise NotImplementedError(f"version {version} not implemented")

    staging_path = f"gs://ml4cc_data_lake/{prod_dev}/1_Staging/WorldFloods"
    train_test_split_file = pkg_resources.resource_filename(
        "ml4floods", f"data/configuration/train_test_split_{dataset}_dataset.json"
    )
    fs_ml4cc = fsspec.filesystem("gs", requester_pays=True)
    files_metadata_pickled = [
        f"gs://{f}"
        for f in fs_ml4cc.glob(f"{staging_path}/*{cems_code}/*{aoi_code}/flood_meta/*.pickle")
    ]

    main_worldlfoods_extra(
        destination_path=destination_parent_path,
        train_test_split_file=train_test_split_file,
        overwrite=overwrite,
        files_metadata_pickled=files_metadata_pickled,
        gt_fun=gt_fun,
        subset=subset,
    )


def main_worldlfoods_extra(
    destination_path: str,
    train_test_split_file: str,
    overwrite: bool = False,
    files_metadata_pickled: list[str] = [],
    gt_fun: Callable = generate_water_cloud_binary_gt,
    subset: str = "all",
):
    """
    Creates the worldfloods_extra dataset in the folder `destination_path`. It copies the files from
    the bucket (from `staging_path`) and creates the ground truth tiff file used for training the models.

    Args:
        destination_path: Path where the dataset will be created
        train_test_split_file: json file with the files used for train, validation, test and baned.
        overwrite: Whether or not to overwrite the files if exists.
        files_metadata_pickled: Path to WorldFloods staging data.
        gt_fun: Function to create the ground truth (3 class ground truth or 2-bands multioutput binary)
        subset: subset to generate the ground truth only for those images
    """

    # Read traintest split file
    fstraintest = utils.get_filesystem(train_test_split_file)
    with fstraintest.open(train_test_split_file, "r") as fh:
        train_val_test_split = json.load(fh)

    cems_codes_test = set(s.split("_")[0] for s in train_val_test_split["test"])
    if "EMSR9284" in cems_codes_test:
        cems_codes_test.add("EMSR284")

    if subset != "all":
        train_val_test_split_new = {subset: train_val_test_split[subset]}
        train_val_test_split = train_val_test_split_new

    skip_unused = subset != "all"

    for s in train_val_test_split:
        print(f"Subset {s} {len(train_val_test_split[s])} ids")

    print(f"Provided {len(files_metadata_pickled)} pickle files to generate the ML-ready dataset")

    # loop through files in the bucket
    problem_files = []
    count_files_per_split = {s: 0 for s in train_val_test_split}
    with tqdm.tqdm(files_metadata_pickled, desc="Generating ground truth extra data") as pbar:
        for metadata_file in pbar:
            metadata_floodmap = utils.read_pickle_from_gcp(metadata_file)
            event_id = metadata_floodmap["event id"]

            # Find out which split to put the data in
            subset_iter = "unused"
            for split in train_val_test_split.keys():
                if (split != "test") and (metadata_floodmap["ems_code"] in cems_codes_test):
                    subset_iter = "banned"

                if event_id in train_val_test_split[split]:
                    subset_iter = split
                    break

            # Do not process if subset is different
            if skip_unused and (subset_iter != subset):
                pbar.write(
                    f"Skipping {metadata_file}. Because it is of {subset_iter} and we are only processing {subset}"
                )
                continue

            # Create destination folder if it doesn't exists
            path_write = os.path.join(destination_path, subset_iter).replace("\\", "/")
            if not path_write.startswith("gs:") and not os.path.exists(path_write):
                os.makedirs(path_write)

            status = generate_item(
                metadata_file,
                path_write,
                file_name=event_id,
                overwrite=overwrite,
                pbar=pbar,
                gt_fun=gt_fun,
                paths_function=worldfloods_extra_gcp_paths,
            )
            if status:
                if subset_iter in count_files_per_split:
                    count_files_per_split[subset_iter] += 1
            else:
                problem_files.append(metadata_file)

    for s in train_val_test_split:
        print(
            f"Split {s} expected {len(train_val_test_split[s])} copied {count_files_per_split[s]}"
        )

    if len(problem_files) > 0:
        print("Files not generated that were expected:")
        for p in problem_files:
            print(p)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Generate WorldFloods ML Dataset")
    parser.add_argument(
        "--version",
        default="v1_0",
        choices=["v1_0", "v2_0"],
        help="Which version of the ground truth we want to create (3-class) or multioutput binary",
    )
    parser.add_argument(
        "--dataset",
        default="extra",
        choices=["original", "extra"],
        help="Use the old data with the new pre-processing 'original' data or "
        "the newly downloaded data from Copernicus EMS with new pre-processing 'extra'",
    )
    parser.add_argument(
        "--prod_dev",
        default="0_DEV",
        choices=["0_DEV", "2_PROD"],
        help="environment where the dataset would be created",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action="store_true",
        help="Overwrite the content in the folder {prod_dev}/2_Mart/worldfloods_{version}",
    )
    parser.add_argument(
        "--cems_code",
        default="",
        help="CEMS Code to download images from. If empty string (default) download the images"
        "from all the codes",
    )
    parser.add_argument(
        "--aoi_code",
        default="",
        help="CEMS AoI to download images from. If empty string (default) download the images"
        "from all the AoIs",
    )
    parser.add_argument(
        "--subset",
        default="all",
        choices=["all", "train", "test", "val"],
        help="all/train/test/val subset to generate the ground truth only for those images"
        "Defaults to all",
    )
    parser.add_argument("--destination_path", default="", help="Destination path")

    args = parser.parse_args()

    main(
        version=args.version,
        overwrite=args.overwrite,
        prod_dev=args.prod_dev,
        dataset=args.dataset,
        cems_code=args.cems_code,
        aoi_code=args.aoi_code,
        destination_parent_path=args.destination_path,
        subset=args.subset,
    )
