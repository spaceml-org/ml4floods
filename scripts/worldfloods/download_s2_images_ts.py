import math
from ml4floods.data import ee_download, utils
import fsspec
from datetime import timedelta, datetime
import os
import pandas as pd
import subprocess
import tempfile
import warnings
import traceback
import sys


def convert_wgs_to_utm(lon: float, lat: float) -> str:
    """Based on lat and lng, return best utm epsg-code"""
    # https://gis.stackexchange.com/questions/269518/auto-select-suitable-utm-zone-based-on-grid-intersection
    # https://stackoverflow.com/questions/40132542/get-a-cartesian-projection-accurate-around-a-lat-lng-pair/40140326#40140326
    utm_band = str((math.floor((lon + 180) / 6 ) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0'+utm_band
    if lat >= 0:
        epsg_code = 'EPSG:326' + utm_band
        return epsg_code
    epsg_code = 'EPSG:327' + utm_band
    return epsg_code


def check_rerun(name_dest_csv, fs, folder_dest, threshold_clouds, threshold_invalids):
    """ Check if any S2 image is missing to trigger download """
    if not fs.exists(name_dest_csv):
        return True

    data = pd.read_csv(name_dest_csv)
    filter_good = (data["cloud_probability"] <= threshold_clouds) & (data["valids"] > (1 - threshold_invalids))
    data = data[filter_good]
    
    if data.shape[0] <= 0:
        return False

    data["datetime"] = data["system:time_start"].apply(lambda x: datetime.utcfromtimestamp(x / 1000))
    for i in range(data.shape[0]):
        date = data['datetime'].iloc[i].strftime('%Y-%m-%d')
        filename = os.path.join(folder_dest, date + ".tif")
        if not fs.exists(filename):
            print(f"Missing files for product {name_dest_csv}. Re-run")
            return True
    return False


def main(cems_code):
    fs = fsspec.filesystem("gs")
    files_metatada_pickled = [f"gs://{f}" for f in fs.glob(f"gs://ml4cc_data_lake/0_DEV/1_Staging/WorldFloods/*{cems_code}/*/flood_meta/*.pickle")]
    THRESHOLD_INVALIDS = .70
    THRESHOLD_CLOUDS = .95
    DAYS_ADD = 20
    DAYS_SUBTRACT = 20
    COLLECTION_NAME = "COPERNICUS/S2" # "COPERNICUS/S2_SR" for atmospherically corrected data

    tasks = []
    for _i, meta_floodmap_filepath in enumerate(files_metatada_pickled):
        print(f"{_i}/{len(files_metatada_pickled)} processing {meta_floodmap_filepath}")

        try:
            metadata_floodmap = utils.read_pickle_from_gcp(meta_floodmap_filepath)
            satellite_date = datetime.strptime(metadata_floodmap["satellite date"].strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S")
            date_start_search = satellite_date + timedelta(days=-DAYS_ADD)
            date_end_search = satellite_date + timedelta(days=DAYS_SUBTRACT)

            aoi_path = os.path.dirname(os.path.dirname(meta_floodmap_filepath))
            folder_dest = os.path.join(aoi_path, "S2")
            # S2 images will be stored in folder_dest path.
            # We will save a csv with the images queried and the available S2 images for that date
            # basename_csv = f"{date_start_search.strftime('%Y%m%d')}_{date_end_search.strftime('%Y%m%d')}.csv"
            basename_csv = f"{DAYS_ADD}_{DAYS_SUBTRACT}_metadata.csv"
            name_dest_csv = os.path.join(folder_dest, basename_csv)
            pol_scene_id = metadata_floodmap["area_of_interest_polygon"]

            # Set the crs to UTM of the center polygon
            lon, lat = list(pol_scene_id.centroid.coords)[0]
            crs = convert_wgs_to_utm(lon=lon, lat=lat)

            name_task = metadata_floodmap["ems_code"] + "_" + metadata_floodmap["aoi_code"]

            if check_rerun(name_dest_csv, fs, folder_dest, threshold_clouds=THRESHOLD_CLOUDS,
                               threshold_invalids=THRESHOLD_INVALIDS):
                tasks_iter, dataframe_images_s2 = ee_download.download_s2(pol_scene_id, date_start_search=date_start_search,
                                                                          date_end_search=date_end_search,
                                                                          crs=crs, path_bucket=folder_dest,
                                                                          name_task=name_task,
                                                                          threshold_invalid=THRESHOLD_INVALIDS,
                                                                          threshold_clouds=THRESHOLD_CLOUDS,
                                                                          collection_name=COLLECTION_NAME)

                if (dataframe_images_s2 is not None) and dataframe_images_s2.shape[0] > 0:
                    # Create csv and copy to bucket
                    with tempfile.NamedTemporaryFile(mode="w", dir=".", suffix=".csv", prefix=os.path.splitext(basename_csv)[0],
                                                     delete=False, newline='') as fh:
                        dataframe_images_s2.to_csv(fh, index=False)
                        basename_csv_local = fh.name

                    subprocess.run(["gsutil", "-m", "mv", basename_csv_local, name_dest_csv])
                    tasks.extend(tasks_iter)
            else:
                print(f"\tAll S2 data downloaded for product")

            # download permanent water
            folder_dest_permament = os.path.join(aoi_path, "PERMANENTWATERJRC")
            task_permanent = ee_download.download_permanent_water(pol_scene_id, date_search=satellite_date,
                                                                  path_bucket=folder_dest_permament,
                                                                  name_task=name_task, crs=crs)
            if task_permanent is not None:
                tasks.append(task_permanent)

        except Exception:
            warnings.warn(f"Failed")
            traceback.print_exc(file=sys.stdout)

    ee_download.wait_tasks(tasks)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Download Copernicus EMS')
    parser.add_argument('--cems_code', default="",
                        help="CEMS Code to download images from. If empty string (default) download the images"
                             "from all the codes")
    args = parser.parse_args()
    main(args.cems_code)
