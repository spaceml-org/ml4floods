from ml4floods.data import ee_download, utils
from datetime import timedelta, datetime, timezone
import os
import pandas as pd
import warnings
import traceback
import sys
import geopandas as gpd


def main(cems_code:str, path_aois:str,flood_date:datetime,
         aoi_code:str, threshold_clouds_before:float,
         threshold_clouds_after:float, threshold_invalids_before:float,
         threshold_invalids_after:float, days_before:int, days_after:int,
         collection_placeholder:str = "S2", only_one_previous:bool=False,
         margin_pre_search:int=0,
         force_s2cloudless:bool=True,
         metadatas_path:str="gs://ml4cc_data_lake/0_DEV/1_Staging/operational/"):
    """

    Args:
        cems_code:
        path_aois:
        flood_date:
        aoi_code:
        threshold_clouds_before:
        threshold_clouds_after:
        threshold_invalids_before:
        threshold_invalids_after:
        days_before:
        days_after:
        collection_placeholder: S2, Landsat or both
        only_one_previous:
        margin_pre_search:
        force_s2cloudless:
        metadatas_path:

    Returns:

    """
    

    fs = utils.get_filesystem(metadatas_path)

    fs_pathaois = utils.get_filesystem(path_aois)
    assert fs_pathaois.exists(path_aois), f"File {path_aois} not found"

    aois_data = gpd.read_file(path_aois)
    assert "name" in aois_data.columns, f"File {path_aois} must have column 'name'"

    # Filter by aoi if provided
    if (aoi_code is not None) and (aoi_code != ""):
        aois_data = aois_data[aois_data["name"] == aoi_code]
        assert aois_data.shape[0] > 0, f"AoI {aois_data} not found in {path_aois}"

    # path_to_glob = f"gs://ml4cc_data_lake/0_DEV/1_Staging/WorldFloods/*{cems_code}/*{aoi_code}/flood_meta/*.pickle"

    # Set collections to download
    if collection_placeholder == "S2":
        collection_names = ["S2"]
        resolutions_meters = [10]
    elif collection_placeholder == "Landsat":
        collection_names = ["Landsat"]
        resolutions_meters = [30]
    elif collection_placeholder == "both":
        collection_names = ["S2", "Landsat"]
        resolutions_meters = [10, 30]
    else:
        raise NotImplementedError(f"Collection name {collection_placeholder} unknown")

    tasks = []
    for _i, row in enumerate(aois_data.itertuples()):
        try:
            pol_scene_id = row.geometry

            folder_dest = os.path.join(metadatas_path, cems_code, row.name).replace("\\","/")

            # Compute arguments to download the images

            date_start_search = flood_date + timedelta(days=-days_before)
            date_end_search = min(datetime.today().astimezone(timezone.utc),
                                  flood_date + timedelta(days=days_after))

            print(f"{_i + 1}/{aois_data.shape[0]} processing images between {date_start_search.strftime('%Y-%m-%d')} and {date_end_search.strftime('%Y-%m-%d')}")

            # Set the crs to UTM of the center polygon
            lon, lat = list(pol_scene_id.centroid.coords)[0]
            crs = ee_download.convert_wgs_to_utm(lon=lon, lat=lat)

            date_pre_flood = flood_date - timedelta(days=margin_pre_search)

            def filter_images(img_col_info_local:pd.DataFrame) -> pd.Series:
                is_image_same_solar_day = img_col_info_local["datetime"].apply(lambda x: (flood_date - x).total_seconds() / 3600. < 10)
                filter_before = (img_col_info_local["cloud_probability"] <= threshold_clouds_before) & \
                                (img_col_info_local["valids"] > (1 - threshold_invalids_before)) & \
                                (img_col_info_local["datetime"] < date_pre_flood) & \
                                (img_col_info_local["datetime"] >= date_start_search) & \
                                ~is_image_same_solar_day

                if only_one_previous and filter_before.any():
                    max_date = img_col_info_local.loc[filter_before, "datetime"].max()
                    filter_before &= (img_col_info_local["datetime"] == max_date)

                filter_after = (img_col_info_local["cloud_probability"] <= threshold_clouds_after) & \
                               (img_col_info_local["valids"] > (1 - threshold_invalids_after)) & \
                               (img_col_info_local["datetime"] <= date_end_search) & \
                               ((img_col_info_local["datetime"] >= flood_date) | is_image_same_solar_day)
                return filter_before | filter_after

            tasks_iter = []
            basename_task = f"{cems_code}_{row.name}"
            for collection_name_trigger, resolution_meters in zip(collection_names, resolutions_meters):
                folder_dest_satellite = os.path.join(folder_dest, collection_name_trigger)
                name_task = collection_name_trigger + "_" + basename_task
                tasks_iter.extend(ee_download.download_s2l89(pol_scene_id,
                                                             date_start_search=date_start_search,
                                                             date_end_search=date_end_search,
                                                             crs=crs,
                                                             filter_fun=filter_images,
                                                             path_bucket=folder_dest_satellite,
                                                             name_task=name_task,
                                                             force_s2cloudless=force_s2cloudless,
                                                             resolution_meters=resolution_meters,
                                                             collection_name=collection_name_trigger))


            if len(tasks_iter) > 0:
                # Create csv and copy to bucket
                tasks.extend(tasks_iter)
            else:
                print(f"\tAll S2 data downloaded for product")

            # download permanent water
            folder_dest_permament = os.path.join(folder_dest, "PERMANENTWATERJRC")
            task_permanent = ee_download.download_permanent_water(pol_scene_id, date_search=flood_date,
                                                                  path_bucket=folder_dest_permament,
                                                                  name_task="PERMANENTWATERJRC"+basename_task,
                                                                  crs=crs)
            if task_permanent is not None:
                tasks.append(task_permanent)

        except Exception:
            warnings.warn(f"Failed {_i} {row.name}")
            traceback.print_exc(file=sys.stdout)

    ee_download.wait_tasks(tasks)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Download Sentinel-2 and Landsat-8/9 images for floodmaps in Staging')
    parser.add_argument("--path_aois", required=True,
                        help="Path to geojson with aoi definition (column name of the aoi_code should be name)")
    parser.add_argument('--cems_code', required=True,
                        help="CEMS Code to download images from. If empty string (default) download the images"
                             "from all the codes")
    parser.add_argument('--flood_date', required=True, help="Date to download the images (YYYY-mm-dd)")
    parser.add_argument('--aoi_code', default="",
                        help="CEMS AoI to download images from. If empty string (default) download the images"
                             "from all the AoIs")
    parser.add_argument('--only_one_previous', action='store_true')
    parser.add_argument('--noforce_s2cloudless', action='store_true')
    parser.add_argument("--collection_name", choices=["Landsat", "S2", "both"], default="both")
    parser.add_argument("--metadatas_path", default="gs://ml4cc_data_lake/0_DEV/1_Staging/operational/",
                        help="gs://ml4cc_data_lake/0_DEV/1_Staging/operational/ for operational floods")
    parser.add_argument('--threshold_clouds_before', default=.1, type=float,
                        help="Threshold clouds before the event")
    parser.add_argument('--threshold_invalids_before', default=.1, type=float,
                        help="Threshold invalids before the event")
    parser.add_argument('--threshold_clouds_after', default=.95, type=float,
                        help="Threshold clouds after the event")
    parser.add_argument('--threshold_invalids_after', default=.70, type=float,
                        help="Threshold invalids after the event")
    parser.add_argument('--days_before', default=20, type=int,
                        help="Days to search after the event")
    parser.add_argument('--days_after', default=20, type=int,
                        help="Days to search before the event")
    parser.add_argument('--margin_pre_search', default=0, type=int,
                        help="Days to include as margin to search for pre-flood images")

    args = parser.parse_args()
    flood_date = datetime.strptime(args.flood_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    main(args.cems_code, path_aois=args.path_aois,flood_date=flood_date,
         aoi_code=args.aoi_code,
         threshold_clouds_before=args.threshold_clouds_before,
         threshold_clouds_after=args.threshold_clouds_after, threshold_invalids_before=args.threshold_invalids_before,
         threshold_invalids_after=args.threshold_invalids_after, days_before=args.days_before,
         collection_placeholder=args.collection_name, metadatas_path=args.metadatas_path,
         only_one_previous=args.only_one_previous, force_s2cloudless=not args.noforce_s2cloudless,
         margin_pre_search=args.margin_pre_search,
         days_after=args.days_after)

