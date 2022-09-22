import argparse
import numpy as np
import rasterio

from ml4floods.data import utils
from ml4floods.models import postprocess
import os
from datetime import datetime, timedelta
import traceback
import sys
import warnings

warnings.filterwarnings('ignore', 'pandas.Int64Index', FutureWarning)

# Sort by date (name of the file) and satellite
def _key_sort(x):
    date = os.path.splitext(os.path.basename(x))[0]
    satellite = os.path.basename(os.path.dirname(x))
    # Preference of Sentinel over Landsat
    if satellite == "Landsat":
        append = "B"
    else:
        append = "A"
    return date+append


def main(model_output_folder:str, flooding_date_pre:str,
         flooding_date_post_start:str,
         flooding_date_post_end:str, overwrite:bool=False):
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

    model_output_folder = model_output_folder.replace("\\","/")
    if model_output_folder.endswith("/"):
        model_output_folder = model_output_folder[:-1]

    activation_folder = os.path.dirname(os.path.dirname(model_output_folder))

    fs = utils.get_filesystem(model_output_folder)

    path_to_search = os.path.join(model_output_folder, "*", "*.geojson")
    prefix = "gs://" if model_output_folder.startswith("gs://") else ""
    geojsons = sorted([f"{prefix}{f}" for f in fs.glob(os.path.join(model_output_folder, "*", "*.geojson").replace("\\", "/"))])
    assert len(geojsons) > 0, f"No products found in {path_to_search}"

    aois = np.unique([g.split("/")[-4] for g in geojsons]).tolist()
    aois.sort(reverse=True)

    pre_flood_paths, post_flood_paths, prepost_flood_paths = [], [], []
    for _iaoi, aoi in enumerate(aois):
        geojsons_iter = [g for g in geojsons if (f"/{aoi}/" in g)]

        aoi_folder = os.path.join(activation_folder, aoi).replace("\\","/")

        # Output products of the processing
        geojsons_iter.sort(key=_key_sort)

        # Create the names of the output products
        pre_flood_path = os.path.join(aoi_folder, "pre_post_products",
                                      f"preflood_{flooding_date_pre}.geojson" ).replace("\\","/")
        pre_flood_paths.append(pre_flood_path)

        post_flood_path = os.path.join(aoi_folder, "pre_post_products",
                                       f"postflood_{flooding_date_post_start}_{flooding_date_post_end}.geojson").replace("\\", "/")
        post_flood_paths.append(post_flood_path)

        prepost_flood_path = os.path.join(aoi_folder, "pre_post_products",
                                          f"prepostflood_{flooding_date_pre}_{flooding_date_post_start}_{flooding_date_post_end}.geojson").replace("\\", "/")
        prepost_flood_paths.append(prepost_flood_path)

        geojsons_post = [g for g in geojsons_iter if (os.path.splitext(os.path.basename(g))[0] >= flooding_date_post_start) and (os.path.splitext(os.path.basename(g))[0] <= flooding_date_post_end)]

        # Check if all output products exist
        if not overwrite:
            all_processed = all(fs.exists(f) for f in [pre_flood_path, post_flood_path, prepost_flood_path])
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
            # Load vectorized JRC permanent water
            permanent_water_floodmap = postprocess.load_vectorized_permanent_water(aoi_folder)

            # Compute joint postflood map (one map for the whole period)
            if (not overwrite) and fs.exists(post_flood_path):
                best_post_flood_data = utils.read_geojson_from_gcp(post_flood_path)
            else:
                best_post_flood_data = postprocess.get_floodmap_post(geojsons_post)
                # Add permanent water polygons
                if permanent_water_floodmap is not None:
                    best_post_flood_data = postprocess.add_permanent_water_to_floodmap(permanent_water_floodmap,
                                                                                       best_post_flood_data,
                                                                                       water_class="water")

                print(f"\tSaving {post_flood_path}")
                utils.write_geojson_to_gcp(post_flood_path, best_post_flood_data)

            # Get pre-flood floodmap with the lowest cloud coverage and all post-flood maps
            geojsons_pre = [g for g in geojsons_iter if os.path.splitext(os.path.basename(g))[0] < flooding_date_pre]
            if len(geojsons_pre) == 0:
                print(f"\tNo pre-flood image found for aoi:{aoi}")
                continue

            # Compute pre-flood data
            if (not overwrite) and fs.exists(pre_flood_path):
                best_pre_flood_data = utils.read_geojson_from_gcp(pre_flood_path)
            else:
                best_pre_flood_data = postprocess.get_floodmap_pre(geojsons_pre)
                # Add permanent water polygons
                if permanent_water_floodmap is not None:
                    best_pre_flood_data = postprocess.add_permanent_water_to_floodmap(permanent_water_floodmap,
                                                                                      best_pre_flood_data,
                                                                                      water_class="water")

                print(f"\tSaving {pre_flood_path}")
                utils.write_geojson_to_gcp(pre_flood_path, best_pre_flood_data)

            # Compute prepost for each floodmap after the flood
            for geojson_post in geojsons_post:
                filename_out = geojson_post.replace("_vec/", "_vec_prepost/")
                if (not overwrite) and fs.exists(filename_out):
                    continue
                if not filename_out.startswith("gs://"):
                    fs.makedirs(os.path.dirname(filename_out), exist_ok=True)

                floodmap_post_data = utils.read_geojson_from_gcp(geojson_post)
                floodmap_post_data_pre_post = postprocess.compute_pre_post_flood_water(floodmap_post_data,
                                                                                       best_pre_flood_data)
                print(f"\tSaving {filename_out}")
                utils.write_geojson_to_gcp(filename_out, floodmap_post_data_pre_post)

            # Compute difference between pre and post floodmap for the whole period
            if overwrite or not fs.exists(prepost_flood_path):
                prepost_flood_data = postprocess.compute_pre_post_flood_water(best_post_flood_data, best_pre_flood_data)
                print(f"\tSaving {prepost_flood_path}")
                utils.write_geojson_to_gcp(prepost_flood_path, prepost_flood_data)

        except Exception:
            traceback.print_exc(file=sys.stdout)

    if len(aois) == 1:
        print("Only one AOI, we will not compute the spatial aggregation")
        return

    path_aggregated_post = os.path.join(activation_folder, f"postflood_{flooding_date_post_start}_{flooding_date_post_end}.geojson")
    path_aggregated_prepost = os.path.join(activation_folder,
                                           f"prepostflood_{flooding_date_pre}_{flooding_date_post_start}_{flooding_date_post_end}.geojson")

    if overwrite or not fs.exists(path_aggregated_post):
        print("AGGREGATE POSTFLOOD MAPS")
        data_all = postprocess.spatial_aggregation([p for p in post_flood_paths if fs.exists(p)])
        utils.write_geojson_to_gcp(path_aggregated_post, data_all)
        print(f"\t{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Saved: {path_aggregated_post}")
    if overwrite or not fs.exists(path_aggregated_prepost):
        print("AGGREGATE PRE-POSTFLOOD MAPS")
        data_all = postprocess.spatial_aggregation([p for p in prepost_flood_paths if fs.exists(p)])
        utils.write_geojson_to_gcp(path_aggregated_prepost, data_all)
        print(f"\t{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Saved: {path_aggregated_prepost}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Post-processing of ML4Floods output to produce pre/post floodmaps')
    parser.add_argument("--model_output_folder", required=True, help="Path to model output folder to run it for all AoIs:"
                                                                     "gs://ml4cc_data_lake/0_DEV/1_Staging/operational/EMSR570/*/WF2_unet_rbgiswirs_vec"
                                                                     "To run it for a single AoI:"
                                                                     "gs://ml4cc_data_lake/0_DEV/1_Staging/operational/EMSR570/AUTOAOI622/WF2_unet_rbgiswirs_vec")
    parser.add_argument("--flooding_date_post_start",
                        help="Flooding date to consider pre-flood maps (Y-m-d in UTC)",
                        required=True)
    parser.add_argument("--flooding_date_post_end",
                        help="Flooding date to consider pre-flood maps (Y-m-d in UTC)",
                        required=False)
    parser.add_argument("--flooding_date_pre",
                        help="Flooding date to consider pre-flood maps (Y-m-d in UTC). "
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

    main(args.model_output_folder, flooding_date_pre=flooding_date_pre.strftime("%Y-%m-%d"),
         flooding_date_post_start=flooding_date_post_start.strftime("%Y-%m-%d"),
         flooding_date_post_end=flooding_date_post_end.strftime("%Y-%m-%d"), overwrite=args.overwrite)

