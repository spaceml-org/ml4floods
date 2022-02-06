from ml4floods.data import ee_download, utils
import fsspec
from datetime import timedelta, datetime, timezone
import os
import pandas as pd
import warnings
import traceback
import sys


def main(cems_code:str, aoi_code:str, threshold_clouds_before:float,
         threshold_clouds_after:float, threshold_invalids_before:float,
         threshold_invalids_after:float, days_before:int, days_after:int, 
         requester_pays:bool = True):
    

    fs = fsspec.filesystem("gs", requester_pays = requester_pays)
    files_metatada_pickled = [f"gs://{f}" for f in fs.glob(f"gs://ml4cc_data_lake/0_DEV/1_Staging/WorldFloods/*{cems_code}/*{aoi_code}/flood_meta/*.pickle")]
    COLLECTION_NAME = "COPERNICUS/S2" # "COPERNICUS/S2_SR" for atmospherically corrected data

    tasks = []
    for _i, meta_floodmap_filepath in enumerate(files_metatada_pickled):
        print(f"{_i}/{len(files_metatada_pickled)} processing {meta_floodmap_filepath}")

        try:
            metadata_floodmap = utils.read_pickle_from_gcp(meta_floodmap_filepath)
            pol_scene_id = metadata_floodmap["area_of_interest_polygon"]
            name_task = metadata_floodmap["ems_code"] + "_" + metadata_floodmap["aoi_code"]
            satellite_date = datetime.strptime(metadata_floodmap["satellite date"].strftime("%Y-%m-%d %H:%M:%S"),
                                               "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

            folder_dest = os.path.dirname(os.path.dirname(meta_floodmap_filepath))

            # Compute arguments to download the images
            folder_dest_s2 = os.path.join(folder_dest, "S2")

            date_start_search = satellite_date + timedelta(days=-days_before)
            date_end_search = min(datetime.today().astimezone(timezone.utc),
                                  satellite_date + timedelta(days=days_after))

            # Set the crs to UTM of the center polygon
            lon, lat = list(pol_scene_id.centroid.coords)[0]
            crs = ee_download.convert_wgs_to_utm(lon=lon, lat=lat)

            def filter_s2_images(img_col_info_local:pd.DataFrame)->pd.Series:
                is_image_same_solar_day = img_col_info_local["datetime"].apply(lambda x: (satellite_date - x).total_seconds() / 3600. < 10)
                filter_clouds_before = (img_col_info_local["cloud_probability"] <= threshold_clouds_before) & \
                                       (img_col_info_local["valids"] > (1 - threshold_invalids_before)) & \
                                       (img_col_info_local["datetime"] < satellite_date) & \
                                        ~is_image_same_solar_day
                filter_clouds_after = (img_col_info_local["cloud_probability"] <= threshold_clouds_after) & \
                                      (img_col_info_local["valids"] > (1 - threshold_invalids_after)) & \
                                      ((img_col_info_local["datetime"] >= satellite_date) | is_image_same_solar_day)
                return filter_clouds_before | filter_clouds_after

            tasks_iter = ee_download.download_s2(pol_scene_id,
                                                 date_start_search=date_start_search,
                                                 date_end_search=date_end_search,
                                                 crs=crs,
                                                 filter_s2_fun=filter_s2_images,
                                                 path_bucket=folder_dest_s2,
                                                 name_task=name_task,
                                                 collection_name=COLLECTION_NAME)

            if len(tasks_iter) > 0:
                # Create csv and copy to bucket
                tasks.extend(tasks_iter)
            else:
                print(f"\tAll S2 data downloaded for product")

            # download permanent water
            folder_dest_permament = os.path.join(folder_dest, "PERMANENTWATERJRC")
            task_permanent = ee_download.download_permanent_water(pol_scene_id, date_search=satellite_date,
                                                                  path_bucket=folder_dest_permament,
                                                                  name_task="PERMANENTWATERJRC"+name_task,
                                                                  crs=crs)
            if task_permanent is not None:
                tasks.append(task_permanent)

        except Exception:
            warnings.warn(f"Failed {files_metatada_pickled}")
            traceback.print_exc(file=sys.stdout)

    ee_download.wait_tasks(tasks)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Download Copernicus EMS')
    parser.add_argument('--cems_code', default="",
                        help="CEMS Code to download images from. If empty string (default) download the images"
                             "from all the codes")
    parser.add_argument('--aoi_code', default="",
                        help="CEMS AoI to download images from. If empty string (default) download the images"
                             "from all the AoIs")
    parser.add_argument('--threshold_clouds_before', default=.3, type=float,
                        help="Threshold clouds before the event")
    parser.add_argument('--threshold_clouds_after', default=.95, type=float,
                        help="Threshold clouds after the event")
    parser.add_argument('--threshold_invalids_before', default=.3, type=float,
                        help="Threshold invalids before the event")
    parser.add_argument('--threshold_invalids_after', default=.70, type=float,
                        help="Threshold invalids after the event")
    parser.add_argument('--days_before', default=20, type=int,
                        help="Days to search after the event")
    parser.add_argument('--days_after', default=20, type=int,
                        help="Days to search before the event")
    

    args = parser.parse_args()

    main(args.cems_code,aoi_code=args.aoi_code, threshold_clouds_before=args.threshold_clouds_before,
         threshold_clouds_after=args.threshold_clouds_after, threshold_invalids_before=args.threshold_invalids_before,
         threshold_invalids_after=args.threshold_invalids_after, days_before=args.days_before,
         days_after=args.days_after)

    # main('EMSR470','AOI01', threshold_clouds_before=args.threshold_clouds_before,
    #       threshold_clouds_after=args.threshold_clouds_after, threshold_invalids_before=args.threshold_invalids_before,
    #       threshold_invalids_after=args.threshold_invalids_after, days_before=args.days_before,
    #       days_after=args.days_after)
