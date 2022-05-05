import argparse
from shapely.ops import unary_union
import geopandas as gpd
import numpy as np
from ml4floods.data import utils
import pandas as pd
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore', 'GeoSeries.isna', UserWarning)


def compute_cloud_coverage(path_to_file:str) -> float:
    data = utils.read_geojson_from_gcp(path_to_file)
    area_total = data[data["class"] == "area_imaged"].geometry.area.sum()
    area_clouds = data[data["class"] == "cloud"].geometry.area.sum()
    return float(area_clouds / area_total)


def compute_flood_water(floodmap_post_data:gpd.GeoDataFrame, best_pre_flood_data:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if floodmap_post_data.crs != best_pre_flood_data.crs:
        best_pre_flood_data = best_pre_flood_data.to_crs(floodmap_post_data.crs)
    else:
        best_pre_flood_data = best_pre_flood_data.copy()

    pre_flood_water = unary_union(best_pre_flood_data[best_pre_flood_data["class"] == "water"].geometry)
    geoms_flood = floodmap_post_data[floodmap_post_data["class"] == "water"].geometry.apply(
        lambda g: g.difference(pre_flood_water))
    geoms_flood = geoms_flood[~geoms_flood.isna() | geoms_flood.is_empty]
    data_post_flood = gpd.GeoDataFrame(geometry=geoms_flood, crs=floodmap_post_data.crs)
    data_post_flood["class"] = "water-post-flood"
    best_pre_flood_data.loc[best_pre_flood_data["class"] == "water", "class"] = "water-pre-flood"
    return pd.concat([best_pre_flood_data, data_post_flood, floodmap_post_data[floodmap_post_data["class"] != "water"]],
                     ignore_index=True)


def main(model_output_folder:str, flooding_date_pre:str, flooding_date_post:str):
    """

    Args:
        model_output_folder: could be:
         gs://ml4cc_data_lake/0_DEV/1_Staging/operational/EMSR570/*/WF2_unet_rbgiswirs_vec
            or: gs://ml4cc_data_lake/0_DEV/1_Staging/operational/EMSR570/AOI01/WF2_unet_rbgiswirs_vec
    """

    fs = utils.get_filesystem(model_output_folder)

    path_to_search = os.path.join(model_output_folder, "*", "*.geojson")
    prefix = "gs://" if model_output_folder.startswith("gs://") else ""
    geojsons = sorted([f"{prefix}{f}" for f in fs.glob(os.path.join(model_output_folder, "*", "*.geojson").replace("\\", "/"))])
    assert len(geojsons) > 0, f"No products found in {path_to_search}"
    aois = np.unique(sorted([g.split("/")[-4] for g in geojsons]))
    for _iaoi, aoi in enumerate(aois):
        floodmaps_post_aoi = [g for g in geojsons if (f"/{aoi}/" in g) and (os.path.splitext(os.path.basename(g))[0] >= flooding_date_post)]

        all_processed = True
        for floodmap_post in floodmaps_post_aoi:
            filename_out = floodmap_post.replace("_vec/", "_vec_prepost/")
            if not fs.exists(filename_out):
                all_processed = False
                break

        if all_processed:
            continue

        # Get pre-flood floodmap with lowest cloud coverage and all post-flood maps
        best_floodmap_pre = None
        cloud_cover = 1
        print(f"({_iaoi+1}/{len(aois)}) Processing AoI: {aoi}")
        for g in geojsons:
            date_iter = os.path.splitext(os.path.basename(g))[0]

            if (f"/{aoi}/" in g) and (date_iter < flooding_date_pre):
                cc_iter = compute_cloud_coverage(g)
                if cc_iter < cloud_cover:
                    best_floodmap_pre = g
                    cloud_cover = cc_iter

        if best_floodmap_pre is None:
            print(f"\tNo pre-flood image found for aoi:{aoi}")
            continue

        # Iterate over the post-flood maps, add pre-flood water and relabel water as "flooding water"
        best_pre_flood_data = utils.read_geojson_from_gcp(best_floodmap_pre)

        for floodmap_post in floodmaps_post_aoi:
            filename_out = floodmap_post.replace("_vec/", "_vec_prepost/")
            if fs.exists(filename_out):
                continue
            floodmap_post_data = utils.read_geojson_from_gcp(floodmap_post)
            floodmap_post_data_pre_post = compute_flood_water(floodmap_post_data, best_pre_flood_data)
            print(f"\tSaving {filename_out}")
            utils.write_geojson_to_gcp(filename_out, floodmap_post_data_pre_post)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Post-processing of ML4Floods output to produce pre/post floodmaps')
    parser.add_argument("--model_output_folder", required=True, help="Path to model output folder (with sub) e.g."
                                                                     "gs://ml4cc_data_lake/0_DEV/1_Staging/operational/EMSR570/*/WF2_unet_rbgiswirs_vec")
    parser.add_argument("--flooding_date_post",
                        help="Flooding date to consider pre-flood maps (Y-m-d in UTC)",
                        required=True)
    parser.add_argument("--flooding_date_pre",
                        help="Flooding date to consider pre-flood maps (Y-m-d in UTC). "
                             "If not provided one day before post",
                        required=False)
    args = parser.parse_args()
    flooding_date_post = datetime.strptime(args.flooding_date_post, "%Y-%m-%d")
    if not args.flooding_date_pre:
        flooding_date_pre = flooding_date_post - timedelta(days=1)
    else:
        flooding_date_pre = datetime.strptime(args.flooding_date_pre, "%Y-%m-%d")

    main(args.model_output_folder, flooding_date_pre=flooding_date_pre.strftime("%Y-%m-%d"),
         flooding_date_post=flooding_date_post.strftime("%Y-%m-%d"))

