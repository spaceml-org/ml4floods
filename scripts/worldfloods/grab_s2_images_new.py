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


from src.data.utils import (
    remove_gcp_prefix,
    get_files_in_directory_gcp,
    read_pickle_from_gcp,
)
from typing import Tuple
from collections import namedtuple
import tqdm


ActivationFile = namedtuple(
    "ActivationFile", ["directory_aoi", "event_activation", "file_name"]
)

# 1. Get Activation Codes, .csv file
CSV_FILE = "gs://ml4cc_data_lake/0_DEV/0_Raw/WorldFloods/copernicus_ems/copernicus_ems_codes/ems_activations_20150701_20210304.csv"
TABLE_ACTIVATIONS_EMS = pd.read_csv(CSV_FILE, encoding="latin1").set_index("Code")
ESMR_CODES = list(TABLE_ACTIVATIONS_EMS[TABLE_ACTIVATIONS_EMS["has_aoi"] == True].index)

# predefined source directories
BUCKET_NAME = "ml4cc_data_lake"
PARENT_DIR_FLOODS = "0_DEV/1_Staging/WorldFloods/floodmap"
PARENT_DIR_FLOOD_META = "0_DEV/1_Staging/WorldFloods/flood_meta"
ALL_FLOOD_FILES = get_files_in_directory_gcp(BUCKET_NAME, PARENT_DIR_FLOODS)
ALL_FLOOD_META_FILES = get_files_in_directory_gcp(BUCKET_NAME, PARENT_DIR_FLOOD_META)
BANDS_EXPORT = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B9",
    "B10",
    "B11",
    "B12",
    "QA60",
    "probability",
]
# predefined target directories
PARENT_DIR_S2 = "0_DEV/1_Staging/WorldFloods/S2"


def find_leaf_nodes(filepath: str) -> Tuple[str, str]:

    filepath = remove_gcp_prefix(filepath)

    directory_aoi = str(Path(filepath).parents[0].parts[-1])
    event_aoi = str(Path(filepath).parents[1].parts[-1])
    file_name = str(Path(filepath).name)

    return ActivationFile(
        directory_aoi=directory_aoi, event_activation=event_aoi, file_name=file_name
    )


from time import sleep


def main():

    # 2. Loop Through Activation Codes
    with tqdm.tqdm(ESMR_CODES, position=0) as pbar_codes:
        for i_esmr_code in pbar_codes:

            # update progress bar
            pbar_codes.set_description(f"ESMR Code: {i_esmr_code}")

            # 3. Get subdirectories
            i_esmr_files = list(filter(lambda x: i_esmr_code in x, ALL_FLOOD_FILES))

            with tqdm.tqdm(i_esmr_files, position=1) as pbar_files:
                for i_file in pbar_files:

                    # update progress bar
                    #                     pbar_files.set_description(f"AOI: {i_file}")

                    activation_aoi_meta = find_leaf_nodes(i_file)

                    # Load Floodmap geojson
                    floodmap_geojson = "gs://" + str(Path(i_file))
                    floodmap = gpd.read_file(floodmap_geojson)

                    # Load Floodmap geojson
                    meta_floodmap_filepath = "gs://" + str(
                        Path(
                            i_file.replace("/floodmap/", "/flood_meta/")
                            .replace(".geojson", ".pickle")
                            .replace("_floodmap.", "_metadata_floodmap.")
                        )
                    )

                    metadata_floodmap = read_pickle_from_gcp(meta_floodmap_filepath)

                    # initialize the GEE
                    ee.Initialize()

                    bounds_pol = activations.generate_polygon(
                        metadata_floodmap["area_of_interest_polygon"].bounds
                    )
                    pol_2_clip = ee.Geometry.Polygon(bounds_pol)

                    # pol with the real area of interest
                    x, y = metadata_floodmap[
                        "area_of_interest_polygon"
                    ].exterior.coords.xy
                    pol_list = list(zip(x, y))
                    pol = ee.Geometry.Polygon(pol_list)

                    date_event = datetime.utcfromtimestamp(
                        metadata_floodmap["satellite_date"].timestamp()
                    )

                    date_end_search = date_event + timedelta(days=20)

                    img_col = ee_download.get_s2_collection(
                        date_event, date_end_search, pol
                    )

                    n_images_col = img_col.size().getInfo()

                    imgs_list = img_col.toList(n_images_col, 0)

                    img_export = ee.Image(imgs_list.get(1))

                    img_export = (
                        img_export.select(BANDS_EXPORT).toFloat().clip(pol_2_clip)
                    )  # .reproject(crs,scale=10).resample('bicubic') resample cannot be used on composites

                    # TODO in the future, change to export to drive and mount the Google drive in colab!

                    bucket_name = "ml4cc_data_lake"

                    export_task_fun_img = ee_download.export_task_image(
                        bucket=BUCKET_NAME
                    )

                    filename = (
                        Path(PARENT_DIR_S2)
                        .joinpath(activation_aoi_meta.event_activation)
                        .joinpath(activation_aoi_meta.directory_aoi)
                        .joinpath(
                            activation_aoi_meta.file_name.replace(
                                "_floodmap.geojson", ""
                            )
                        )
                    )
                    filename = str(filename)
                    desc = os.path.basename(filename)

                    # DOWNLOAD!!!!
                    task = ee_download.mayberun(
                        filename,
                        desc,
                        lambda: img_export,
                        export_task_fun_img,
                        overwrite=False,
                        dry_run=False,
                        bucket_name=bucket_name,
                        verbose=2,
                    )
                    if task is not None:
                        print(task.status())

    #                     break

    #             break

    pass


if __name__ == "__main__":
    main()
