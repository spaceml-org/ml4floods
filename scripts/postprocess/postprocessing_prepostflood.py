import argparse
import numpy as np
from ml4floods.data import utils
from ml4floods.models import postprocess
import os
from datetime import datetime, timedelta
import traceback
import sys
import warnings
from tqdm import tqdm
import rasterio.warp
from shapely.geometry import shape
import pandas as pd
from typing import List


warnings.filterwarnings('ignore', 'pandas.Int64Index', FutureWarning)


def main(model_output_folder:str, flooding_date_pre:str,
         flooding_date_post_start:str, flooding_date_post_end:str,
         overwrite:bool=False):
    """

    Args:
        model_output_folder:
            'gs://ml4cc_data_lake/0_DEV/1_Staging/operational/EMSR570/*/WF2_unet_rbgiswirs_vec'
            'gs://ml4cc_data_lake/0_DEV/1_Staging/operational/EMSR570/AOI01/WF2_unet_rbgiswirs_vec'
        flooding_date_pre:
        flooding_date_post_start:
        flooding_date_post_end:
        overwrite:

    """

    # Parse the model/activation path to a fsspec object
    model_output_folder = model_output_folder.replace("\\","/")
    if model_output_folder.endswith("/"):
        model_output_folder = model_output_folder[:-1]
    activation_folder = os.path.dirname(os.path.dirname(model_output_folder))
    fs = utils.get_filesystem(model_output_folder)

    # Search for GeoJSON water masks in the model folder
    path_to_search = os.path.join(model_output_folder, "*", "*.geojson")
    prefix = "gs://" if model_output_folder.startswith("gs://") else ""
    geojsons = sorted([f"{prefix}{f}" for f in fs.glob(os.path.join(model_output_folder, "*", "*.geojson").replace("\\", "/"))])
    assert len(geojsons) > 0, f"No products found in {path_to_search}"

    # Parse a unique list of AOI names from the model folder
    aois = np.unique([g.split("/")[-4] for g in geojsons]).tolist()
    aois.sort(reverse=True)

    # Loop through the AOIs and compute the pre- and post-flood water maps, and diff.
    pre_flood_paths, post_flood_paths, prepost_flood_paths = [], [], []
    for _iaoi, aoi in enumerate(aois):

        # List and sort GeoJSONs in the current AOI (by date and satellite)
        geojsons_iter = [g for g in geojsons if (f"/{aoi}/" in g)]
        geojsons_iter.sort(key=_key_sort)

        # Format the lists of output paths
        aoi_folder = os.path.join(activation_folder, aoi).replace("\\","/")
        pre_flood_path = os.path.join(
            aoi_folder, "pre_post_products",
            f"preflood_{flooding_date_pre}.geojson" ).replace("\\","/")
        pre_flood_paths.append(pre_flood_path)
        post_flood_path = os.path.join(
            aoi_folder, "pre_post_products",
            f"postflood_{flooding_date_post_start}_{flooding_date_post_end}.geojson").replace("\\", "/")
        post_flood_paths.append(post_flood_path)
        prepost_flood_path = os.path.join(
            aoi_folder, "pre_post_products",
            f"prepostflood_{flooding_date_pre}_{flooding_date_post_start}_{flooding_date_post_end}.geojson").replace("\\", "/")
        prepost_flood_paths.append(prepost_flood_path)

        # List the GeoJSONs between [post_start, post_end], inclusive
        geojsons_post = [g for g in geojsons_iter
                         if (os.path.splitext(os.path.basename(g))[0]
                             >= flooding_date_post_start)
                         and (os.path.splitext(os.path.basename(g))[0]
                              <= flooding_date_post_end)]

        # Check if all output products exist
        if not overwrite:
            all_processed = all(
                fs.exists(f) for f in [pre_flood_path, post_flood_path, prepost_flood_path])
            if all_processed:
                for geojson_post in geojsons_post:
                    filename_out = geojson_post.replace("_vec/", "_vec_prepost/")
                    if not fs.exists(filename_out):
                        all_processed = False
                        break

            if all_processed:
                continue

        print(f"({_iaoi + 1}/{len(aois)}) Processing AoI: {aoi}")

        if len(geojsons_post) == 0:
            print(f"\tNo post-flood images found for aoi:{aoi}")
            continue

        try:
            # Compute joint post-flood water map (one map for the whole period)
            if (not overwrite) and fs.exists(post_flood_path):
                # Returns the existing joined floodmap
                best_post_flood_data = utils.read_geojson_from_gcp(post_flood_path)
            else:
                # Perform the flood-map merger and write to storage
                best_post_flood_data = postprocess.get_floodmap_post(geojsons_post)
                print(f"\tSaving {post_flood_path}")
                utils.write_geojson_to_gcp(post_flood_path, best_post_flood_data)

            # List the GeoJSONs that cover the pre-flood dates
            geojsons_pre = [g for g in geojsons_iter
                            if os.path.splitext(os.path.basename(g))[0] < flooding_date_pre]
            if len(geojsons_pre) == 0:
                print(f"\tNo pre-flood image found for aoi:{aoi}")
                continue

            # Compute the joint pre-flood water maps
            if (not overwrite) and fs.exists(pre_flood_path):
                best_pre_flood_data = utils.read_geojson_from_gcp(pre_flood_path)
            else:
                best_pre_flood_data = postprocess.get_floodmap_pre(geojsons_pre)
                print(f"\tSaving {pre_flood_path}")
                utils.write_geojson_to_gcp(pre_flood_path, best_pre_flood_data)

            # Compute the joint map of the flood water (post-flood - pre-flood mask)
            for geojson_post in geojsons_post:
                filename_out = geojson_post.replace("_vec/", "_vec_prepost/")
                if (not overwrite) and fs.exists(filename_out):
                    continue
                if not filename_out.startswith("gs://"):
                    fs.makedirs(os.path.dirname(filename_out), exist_ok=True)
                floodmap_post_data = utils.read_geojson_from_gcp(geojson_post)
                floodmap_post_data_pre_post = \
                    postprocess.compute_flood_water(floodmap_post_data, best_pre_flood_data)
                floodmap_post_data_pre_post["id"] = \
                    np.arange(0, floodmap_post_data_pre_post.shape[0])
                print(f"\tSaving {filename_out}")
                utils.write_geojson_to_gcp(filename_out, floodmap_post_data_pre_post)

            # Compute difference between pre and post floodmap for the whole period
            if overwrite or not fs.exists(prepost_flood_path):
                prepost_flood_data = postprocess.compute_flood_water(best_post_flood_data,
                                                                     best_pre_flood_data)
                print(f"\tSaving {prepost_flood_path}")
                utils.write_geojson_to_gcp(prepost_flood_path, prepost_flood_data)

        except Exception:
            traceback.print_exc(file=sys.stdout)

    # Construct the paths for the final spatially aggregated maps
    path_aggregated_post = os.path.join(activation_folder, f"postflood_{flooding_date_post_start}_{flooding_date_post_end}.geojson")
    path_aggregated_prepost = os.path.join(activation_folder, f"prepostflood_{flooding_date_pre}_{flooding_date_post_start}_{flooding_date_post_end}.geojson")

    # Perform the spatial aggregation and write to storage
    if overwrite or not fs.exists(path_aggregated_post):
        print("AGGREGATE POSTFLOOD MAPS")
        spatial_aggregation(path_aggregated_post,
                            [p for p in post_flood_paths if fs.exists(p)])
    if overwrite or not fs.exists(path_aggregated_prepost):
        print("AGGREGATE PRE-POSTFLOOD MAPS")
        spatial_aggregation(path_aggregated_prepost,
                            [p for p in prepost_flood_paths if fs.exists(p)])


def _key_sort(x):
    """
    Use the date component of a filename as a sort key and sort
    Sentinel-2 before Landsat by appending A and B, respectively.
    """
    date = os.path.splitext(os.path.basename(x))[0]
    satellite = os.path.basename(os.path.dirname(x))
    if satellite == "Landsat":
        append = "B"
    else:
        append = "A"
    return date + append


def spatial_aggregation(path_aggregated_post:str, floodmaps_paths:List[str],
                        dst_crs:str= "EPSG:4326"):
    """
    Join the list of floodmaps from different AoIs into a single one.
    """

    data_all = None

    for f in tqdm(floodmaps_paths):
        # Read in flood masks and filter for NaNs, empties and small polygons
        data = utils.read_geojson_from_gcp(f)
        data = data[~data.geometry.isna() &
                    ~data.geometry.is_empty &
                    (data.geometry.area > 10 ** 2)].copy()
        # Convert to CRS with rasterio!
        geometry = rasterio.warp.transform_geom(
            src_crs=data.crs,
            dst_crs=dst_crs,
            geom=data.geometry.values,
        )
        data = data.set_geometry(
            [shape(geom) for geom in geometry],
            crs=dst_crs,
        )
        is_valid_geoms = data.is_valid
        if not is_valid_geoms.all():
            print(f"\tThere are still {(~is_valid_geoms).sum()} geoms invalid of {is_valid_geoms.shape[0]}")
            data = data[is_valid_geoms]

        if data_all is None:
            data_all = data
        else:
            data_all = pd.concat([data_all, data], ignore_index=True)
    print(f"\t{len(floodmaps_paths)} Products joined {data_all.shape}")
    # Save as geojson
    data_all = data_all.dissolve(by="class").reset_index()
    print(f"\t{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Correctly dissolved. New shape: {data_all.shape}")
    data_all = data_all.explode(ignore_index=True)
    print(f"\t{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Correctly exploded. New shape: {data_all.shape}")
    data_all = data_all[~data_all.geometry.isna() & ~data_all.geometry.is_empty]
    utils.write_geojson_to_gcp(path_aggregated_post, data_all)
    print(f"\t{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Saved: {path_aggregated_post}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'Post-processing of ML4Floods output to produce pre/post floodmaps')
    parser.add_argument("--model_output_folder", required=True,
                        help="Path to model output folder (with sub) e.g."
                        "gs://ml4cc_data_lake/0_DEV/1_Staging/operational/EMSR570/*/WF2_unet_rbgiswirs_vec")
    parser.add_argument("--flooding_date_post_start",
                        help="Flooding start date (Y-m-d in UTC)",
                        required=True)
    parser.add_argument("--flooding_date_post_end",
                        help="Flooding end date (Y-m-d in UTC)",
                        required=False)
    parser.add_argument("--flooding_date_pre",
                        help="Flooding start date for pre-flood maps (Y-m-d in UTC). "
                             "If not provided one day before post",
                        required=False)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    flooding_date_post_start = datetime.strptime(args.flooding_date_post_start, "%Y-%m-%d")
    if not args.flooding_date_pre:
        flooding_date_pre = flooding_date_post_start - timedelta(days=1)
    else:
        flooding_date_pre = datetime.strptime(args.flooding_date_pre, "%Y-%m-%d")

    if not args.flooding_date_post_end:
        flooding_date_post_end = flooding_date_post_start + timedelta(days=20)
    else:
        flooding_date_post_end = datetime.strptime(args.flooding_date_post_end, "%Y-%m-%d")

    main(args.model_output_folder,
         flooding_date_pre=flooding_date_pre.strftime("%Y-%m-%d"),
         flooding_date_post_start=flooding_date_post_start.strftime("%Y-%m-%d"),
         flooding_date_post_end=flooding_date_post_end.strftime("%Y-%m-%d"),
         overwrite=args.overwrite)
