import json
import logging
import os
import re
from glob import glob

import geopandas as gpd
import rasterio

from ml4floods.data import create_gt


def get_layer_name(tiffile:str) -> str:
    """
    The GEE splits large areas in different tifffiles.
    This function gets the name of the floodmap file
    """
    layer_name = os.path.splitext(os.path.basename(tiffile))[0]
    if re.search("\d{10}-\d{10}", layer_name) is not None:
        layer_name = layer_name[:-21]
    return layer_name


def save_cog(out_np, path_tiff_save, profile, tags=None):
    """
    saves `out_np` np array as a COG GeoTIFF in path_tiff_save. profile is a dict with the geospatial info to be saved
    with the TiFF.

    :param out_np: 3D numpy array to save in CHW format
    :param path_tiff_save:
    :param profile: dict with profile to write geospatial info of the dataset: (crs, transform)
    :param tags: extra info to save as tags
    """

    # Set count, height, width
    for idx, c in enumerate(["count", "height", "width"]):
        if c in profile:
            assert profile[c] == out_np.shape[idx], f"Unexpected shape: {profile[c]} {out_np.shape}"
        else:
            profile[c] = out_np.shape[idx]

    for field in ["crs","transform"]:
        assert field in profile, f"{field} not in profile. it will not write cog without geo information"

    profile["BIGTIFF"] = "IF_SAFER"
    with rasterio.Env() as env:
        cog_driver =  "COG" in env.drivers()

    assert cog_driver, f"COG driver not installed update to gdal>=3.1"

    profile["driver"] = "COG"
    with rasterio.open(path_tiff_save, "w", **profile) as rst_out:
        if tags is not None:
            rst_out.update_tags(**tags)
        rst_out.write(out_np)
    return path_tiff_save


if __name__ == "__main__":

    generate_gt_fun = create_gt.generate_gt_v2
    gtversion = "v2"

    # Read floodmaps metadata
    floodmaps_metadata = dict()
    for f in glob(os.path.join("worldfloods/floodmaps/meta/*.json")):
        with open(f,"r") as fh:
            floodmap_name = os.path.splitext(os.path.basename(f))[0]
            floodmaps_metadata[floodmap_name] = json.load(fh)
            floodmaps_metadata["filename"] = f

    # TODO glob in the bucket?
    s2files = sorted(glob(os.path.join("worldfloods/tiffimages/S2/*.tif")))
    s2_files = s2files[50:] + s2files[:50]  # Trick to process first the small files (for recomputing)

    for _i, s2_tiff_path in enumerate(s2files):
        name_s2file = os.path.splitext(os.path.basename(s2_tiff_path))[0]
        logging.info("%d/%d %s" % (_i + 1, len(s2files), s2_tiff_path))
        layer_name = get_layer_name(s2_tiff_path)
        if layer_name not in floodmaps_metadata:
            logging.warning("\t %s (%s) not found in vector files!" % (s2_tiff_path, layer_name))
            continue

        floodmap_metadata = floodmaps_metadata[layer_name]

        floodmap_file = floodmaps_metadata["filename"].replace("/meta/","/floodmap/")
        assert os.path.exists(floodmap_file), \
            f"{floodmap_file} not found for corresponding metadata file {floodmap_metadata}"

        floodmap = gpd.read_file(floodmap_file)

        # If it does not exists if will compute the cloud mask
        cloudprob_file = s2_tiff_path.replace("/S2/","/cloudprob/")
        if not os.path.exists(cloudprob_file):
            cloudprob_file = None

        # It could happen that it does not exists for this particular file raise warning?
        pernament_water_file = s2_tiff_path.replace("/S2/", "/PERMANENTWATERJRC/")
        if not os.path.exists(pernament_water_file):
            logging.warning(f"{pernament_water_file} not found. {name_s2file} will not use permanent water to create GT")
            pernament_water_file = None

        gt, metadata_gt = generate_gt_fun(s2tiff=s2_tiff_path, floodmap=floodmap, metadata_floodmap=floodmap_metadata,
                                          window=None,permanent_water_tiff=pernament_water_file, cloudprob_tiff=cloudprob_file,
                                          cloudprob_in_lastband=False)

        # Save gt and metadata_gt
        with rasterio.open(s2_tiff_path) as src:
            dst_crs = src.crs
            transform = src.transform

        profile = {
            "dtype": rasterio.uint8,
            "crs": dst_crs,
            "nodata": 0,
            "compress": "lzw",
            "RESAMPLING": "NEAREST",  # for pyramids
            "transform": transform,
        }
        save_cog(gt, f"worldfloods/tiffimages/gt{gtversion}/{name_s2file}.tif", profile=profile)
        with open(f"worldfloods/tiffimages/meta{gtversion}/{name_s2file}.json", "w") as fh:
            json.dump(metadata_gt, fh)





