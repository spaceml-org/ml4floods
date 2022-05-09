import argparse
import numpy as np
from ml4floods.data import utils
from ml4floods.models import postprocess
import os
from datetime import datetime, timedelta


def main(model_output_folder:str, flooding_date_pre:str, flooding_date_post:str, overwrite:bool=False):
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
        geojsons_iter = [g for g in geojsons if (f"/{aoi}/" in g)]

        # Do not compute if not needed
        floodmaps_post_aoi = [g for g in geojsons_iter if (os.path.splitext(os.path.basename(g))[0] >= flooding_date_post)]
        if not overwrite:
            all_processed = True
            for floodmap_post in floodmaps_post_aoi:
                filename_out = floodmap_post.replace("_vec/", "_vec_prepost/")
                if not fs.exists(filename_out):
                    all_processed = False
                    break

            if all_processed:
                continue

        print(f"({_iaoi + 1}/{len(aois)}) Processing AoI: {aoi}")

        # Get pre-flood floodmap with lowest cloud coverage and all post-flood maps
        best_pre_flood_data = postprocess.get_floodmap_pre(flooding_date_pre, geojsons_iter)

        if best_pre_flood_data is None:
            print(f"\tNo pre-flood image found for aoi:{aoi}")
            continue

        for floodmap_post in floodmaps_post_aoi:
            filename_out = floodmap_post.replace("_vec/", "_vec_prepost/")
            if (not overwrite) and fs.exists(filename_out):
                continue
            if not filename_out.startswith("gs://"):
                fs.makedirs(os.path.dirname(filename_out), exist_ok=True)

            floodmap_post_data = utils.read_geojson_from_gcp(floodmap_post)
            floodmap_post_data_pre_post = postprocess.compute_flood_water(floodmap_post_data, best_pre_flood_data)
            floodmap_post_data_pre_post["id"] = np.arange(0, floodmap_post_data_pre_post.shape[0])
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
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    flooding_date_post = datetime.strptime(args.flooding_date_post, "%Y-%m-%d")
    if not args.flooding_date_pre:
        flooding_date_pre = flooding_date_post - timedelta(days=1)
    else:
        flooding_date_pre = datetime.strptime(args.flooding_date_pre, "%Y-%m-%d")

    main(args.model_output_folder, flooding_date_pre=flooding_date_pre.strftime("%Y-%m-%d"),
         flooding_date_post=flooding_date_post.strftime("%Y-%m-%d"),overwrite=args.overwrite)

