import sys
import warnings
import traceback

import tqdm
import pandas as pd
from datetime import datetime
from ml4floods.data import utils, vectorize
from ml4floods.data.ee_download import process_metadata
import os
import numpy as np

from pathlib import Path
from typing import Optional, Callable, Tuple, Dict, Any
import geopandas as gpd
import fsspec
from ml4floods.data import save_cog


CLOUDPROB_PARENT_PATH = "worldfloods/tiffimages"
PERMANENT_WATER_PARENT_PATH = "worldfloods/tiffimages/PERMANENTWATERJRC"
META_FLOODMAP_PARENT_PATH = "worldfloods/tiffimages/meta"
WORLDFLOODS_V0_BUCKET = "ml4floods"


def worldfloods_extra_gcp_paths(main_path: str) -> Tuple[gpd.GeoDataFrame, Optional[str], Optional[str], Dict, str]:
    """
    Given a pickle file in "gs://ml4cc_data_lake/{prod_dev}/1_Staging/WorldFloods it returns the rest of the files to
    create the full worldfloods registry (the corresponding floodmap, cloud probability and permanent water)

    Args:
        main_path: path to .piclke file in bucket

    Returns:
        Locations of corresponding  floodmap, cloud probability,  permanent water, metadata_floodmap and S2 image

    """
    fs = fsspec.filesystem("gs", requester_pays=True)
    assert fs.exists(main_path), f"File {main_path} does not exists"

    meta_floodmap = utils.read_pickle_from_gcp(main_path)

    # path to floodmap path
    floodmap_path = main_path.replace("/flood_meta/", "/floodmap_edited/").replace(".pickle", ".geojson")
    if not fs.exists(floodmap_path):
        floodmap_path = floodmap_path.replace("/floodmap_edited/", "/floodmap/")

    assert fs.exists(floodmap_path), f"Floodmap not found in {floodmap_path}"

    path_aoi = os.path.dirname(os.path.dirname(floodmap_path))

    # open floodmap with geopandas
    floodmap = utils.read_geojson_from_gcp(floodmap_path)
    floodmap_date = meta_floodmap['satellite date']

    # create permenant water path
    permanent_water_path = os.path.join(path_aoi, "PERMANENTWATERJRC", f"{floodmap_date.year}.tif").replace("\\", "/")

    if not fs.exists(permanent_water_path):
        warnings.warn(f"Permanent water {permanent_water_path}. Will not be used")
        permanent_water_path = None

    csv_path = os.path.join(path_aoi, "S2", "s2info.csv")
    metadatas2 = process_metadata(csv_path, fs=fs)
    metadatas2 = metadatas2.set_index("names2file")

    assert any(metadatas2.s2available), f"Not available S2 files for {main_path}. {metadatas2}"

    # find corresponding s2_image_path for this image
    index, s2_date = best_s2_match(metadatas2, floodmap_date)

    assert s2_date is not None, f"Not found valid S2 files for {main_path}. {metadatas2}"

    assert (s2_date - floodmap_date).total_seconds() / (3600. * 24) < 10, \
        f"Difference between S2 date {s2_date} and floodmap date {floodmap_date} is larger than 10 days"

    # add exact metadata from the csv file
    meta_floodmap["s2_date"] = s2_date
    meta_floodmap["names2file"] = index
    meta_floodmap["cloud_probability"] = metadatas2.loc[index, "cloud_probability"]
    meta_floodmap["valids"] = metadatas2.loc[index, "valids"]

    s2_image_path = os.path.join(path_aoi, "S2", index+".tif").replace("\\", "/")

    # Add cloud_probability if exists in edited
    cm_edited = s2_image_path.replace("/S2/", "/cmedited_vec/").replace(".tif", ".geojson")
    if not fs.exists(cm_edited):
        cm_edited = s2_image_path.replace("/S2/", "/cmedited/")
        if not fs.exists(cm_edited):
            cm_edited = None

    return floodmap, cm_edited, permanent_water_path, meta_floodmap, s2_image_path


def best_s2_match(metadatas2:pd.DataFrame, floodmap_date:datetime) -> Tuple[Any, datetime]:
    """
    Return s2 date posterior to the floodmap_date

    Args:
        metadatas2:
        floodmap_date:

    Returns:

    """
    index = None
    s2_date = None
    for tup in metadatas2[metadatas2.s2available].itertuples():
        date_img = tup.datetime
        if (floodmap_date < date_img) or ((floodmap_date - date_img).total_seconds() / 3600. < 10):
            if s2_date is None:
                s2_date = date_img
                index = tup.Index
            else:
                if s2_date > date_img:
                    s2_date = date_img
                    index = tup.Index
    return index, s2_date


def worldfloods_old_gcp_paths(main_path: str) -> Tuple[gpd.GeoDataFrame, str, Optional[str], Dict, str]:
    """
    Given a S2 tiff file in the V0 WorldFloods dataset it returns the rest of the anciliary files to
    the corresponding floodmap, cloud probability and permanent water

    Args:
        main_path: S2 path in gs://ml4floods/worldfloods/public folder

    Returns:
        GCPPaths with locations of corresponding  floodmap, cloud probability,  permanent water, meta_floodmap and sentinel2 image

    Examples
        >>> s2_path_file = "gs://ml4floods/worldfloods/public/train/S2/EMSR260_09RUBIERA_GRA_v2_observed_event_a.tif"
        >>> worldfloods_old_gcp_paths(GCPPath(s2_path_file))

    (GCPPath(full_path='gs://ml4floods/worldfloods/public/train/floodmaps/EMSR260_09RUBIERA_GRA_v2_observed_event_a.shp', bucket_id='ml4floods', parent_path='worldfloods/public/train/floodmaps', file_name='EMSR260_09RUBIERA_GRA_v2_observed_event_a.shp', suffix='shp'),
     GCPPath(full_path='gs://ml4floods/worldfloods/tiffimages/cloudprob/EMSR260_09RUBIERA_GRA_v2_observed_event_a.tif', bucket_id='ml4floods', parent_path='worldfloods/tiffimages/cloudprob', file_name='EMSR260_09RUBIERA_GRA_v2_observed_event_a.tif', suffix='tif'),
     'gs://ml4floods/worldfloods/tiffimages/PERMANENTWATERJRC/EMSR260_09RUBIERA_GRA_v2_observed_event_a.tif')

    """
    assert main_path.check_if_file_exists(), f"File {main_path} does not exists"

    s2_image_path = main_path

    # path to floodmap path
    floodmap_path = s2_image_path.replace("/S2/", "/floodmaps/")
    floodmap_path = floodmap_path.replace(".tif", f".shp")

    assert floodmap_path.check_if_file_exists(), f"Floodmap not found in {floodmap_path}"

    # open floodmap with geopandas
    floodmap = gpd.read_file(floodmap_path.full_path)

    # create cloudprob path
    cloudprob_path = GCPPath(
        str(
            Path(WORLDFLOODS_V0_BUCKET)
                .joinpath(CLOUDPROB_PARENT_PATH)
                .joinpath("cloudprob_edited")
                .joinpath(s2_image_path.file_name)
        )
    )
    if not cloudprob_path.check_if_file_exists():
        cloudprob_path = GCPPath(
            str(
                Path(WORLDFLOODS_V0_BUCKET)
                    .joinpath(CLOUDPROB_PARENT_PATH)
                    .joinpath("cloudprob")
                    .joinpath(s2_image_path.file_name)
            )
        )

    assert cloudprob_path.check_if_file_exists(), f"Clouds not found in {cloudprob_path}"

    # create permenant water path
    permanent_water_path = GCPPath(
        str(
            Path(WORLDFLOODS_V0_BUCKET)
                .joinpath(PERMANENT_WATER_PARENT_PATH)
                .joinpath(s2_image_path.file_name)
        )
    )

    if not permanent_water_path.check_if_file_exists():
        warnings.warn(f"Permanent water {permanent_water_path}. Will not be used")
        permanent_water_path = None


    # path to meta_floodmap
    meta_floodmap_path = GCPPath(
        str(
            Path(WORLDFLOODS_V0_BUCKET)
                .joinpath(META_FLOODMAP_PARENT_PATH)
                .joinpath(s2_image_path.file_name.replace(".tif",".json"))
        )
    )
    assert meta_floodmap_path.check_if_file_exists(), f"Meta floodmap not found in {meta_floodmap_path}"

    meta_floodmap = utils.read_json_from_gcp(meta_floodmap_path.full_path)

    return floodmap, cloudprob_path, permanent_water_path, meta_floodmap, s2_image_path


def generate_item(main_path:str, output_path:str, file_name:str,
                  overwrite:bool=False, pbar:Optional[tqdm.tqdm]=None,
                  gt_fun:Callable=None, delete_if_error:bool=True,
                  paths_function:Callable=worldfloods_old_gcp_paths) -> bool:
    """

    Generates an "element" of the WorldFloods dataset with the expected naming convention. An "element" is a set of files
    that will be used for training the WorldFloods model. These files are copied in the `output_path` folowing the naming
    convention coded in `worldfloods_output_files` function.
     These files are:
    - shp vector floodmap. (subfolder floodmap)
    - tiff permanent water (subfolder PERMANENTWATERJRC)
    - tiff gt
    - tiff S2
    - tiff cloudprob
    - json with metadata info of the ground truth

    Args:
        main_path: Path to main object. The process will search for other relevant information to create all the
        aforementioned products.
        output_path: Folder where the item will be written. See fun worldfloods_output_files for corresponding output file naming convention.
        overwrite: if False it will not overwrite existing products in path_write folder.
        file_name: Name of the file to be saved (e.g. in S2/gt/floodmap/ data will be saved with this name and the corresponding extension .tif, .geojson)
        pbar: optional progress bar with method description.
        gt_fun: one of ml4floods.data.create_gt.generate_land_water_cloud_gt or ml4floods.data.create_gt.generate_water_cloud_binary_gt.
        This function determines how the ground truth is created from the input products.
        delete_if_error: whether to delete the generated files if an error is risen
        paths_function: function to get the paths of the files for the given main_path

    Returns:
        True if success in creating all the products

    """

    fs = fsspec.filesystem("gs", requester_pays=True)

    try:
        # Check if output products exist before reading from the bucket
        if not overwrite:
            expected_outputs = worldfloods_output_files(
                output_path, file_name, permanent_water_available=True, clouds_available=False, mkdirs=False)
            fsdest = utils.get_filesystem(expected_outputs[-1])

            must_process = False
            for e in expected_outputs:
                if e and not fsdest.exists(e):
                    must_process = True
                    break

            if not must_process:
                return True

        # Get input files and check that they all exist
        floodmap, cloudprob_path, permanent_water_path, metadata_floodmap, s2_image_path = paths_function(main_path)

        # get output files
        cloudprob_path_dest, floodmap_path_dest, gt_path_dest, meta_json_path_dest, permanent_water_image_path_dest, s2_image_path_dest = worldfloods_output_files(
            output_path, file_name, permanent_water_available=permanent_water_path is not None, clouds_available=cloudprob_path is not None, mkdirs=True)
    except Exception:
        warnings.warn(f"File {main_path} problem when computing input/output names")
        traceback.print_exc(file=sys.stdout)
        return False
    try:
        # generate gt, gt meta and copy all files to path_write
        fsdest = utils.get_filesystem(s2_image_path_dest)

        if not fsdest.exists(gt_path_dest) or not fsdest.exists(meta_json_path_dest) or overwrite:
            if pbar is not None:
                pbar.write(f"Generating Ground Truth {file_name}...")

            # Copy s2_image_path to local before reading?
            # If so we will need also to copy cloudprob_path and permanent_water_path

            gt, gt_meta = gt_fun(
                s2_image_path,
                floodmap,
                metadata_floodmap=metadata_floodmap,
                keep_streams=True,
                cloudprob_image_path=cloudprob_path, # Could be None!
                permanent_water_image_path=permanent_water_path,  # Could be None!
            )

            if len(gt.shape) == 2:
                gt = gt[None]

            if pbar is not None:
                pbar.write(f"Saving GT {file_name}...")

            if gt.shape[0] == 2:
                desc = ["invalid/clear/cloud", "invalid/land/water"]
            else:
                desc = ["invalid/land/water/cloud"]

            save_cog.save_cog(gt, gt_path_dest,
                              {"crs": gt_meta["crs"], "transform":gt_meta["transform"] ,"RESAMPLING": "NEAREST",
                               "compress": "lzw", "nodata": 0}, # In both gts 0 is nodata
                              descriptions=desc,
                              tags=gt_meta)

            # upload meta json to bucket
            if pbar is not None:
                pbar.write(f"Saving meta {file_name}...")

            # save meta in local json file
            gt_meta["crs"] = str(gt_meta["crs"])
            gt_meta["transform"] = [gt_meta["transform"].a, gt_meta["transform"].b, gt_meta["transform"].c,
                                    gt_meta["transform"].d, gt_meta["transform"].e, gt_meta["transform"].f]

            utils.write_json_to_gcp(meta_json_path_dest, gt_meta)

        # Copy cloudprob
        if cloudprob_path_dest and (not fsdest.exists(cloudprob_path_dest) or overwrite):
            if pbar is not None:
                pbar.write(f"Saving cloud probs {file_name}...")

            with utils.rasterio_open_read(gt_path_dest) as rst:
                gt = rst.read()
                transform = rst.transform
                crs = rst.crs

            if gt.shape[0] == 2:
                clouds = gt[0] == 2
            else:
                clouds = gt[0] == 3

            # vectorize clouds
            geoms_polygons = vectorize.get_polygons(clouds,
                                                    transform=transform)
            if len(geoms_polygons) > 0:
                clouds_vec = gpd.GeoDataFrame({"geometry": geoms_polygons,
                                               "class": "cloud"},
                                              crs=crs)
            else:
                clouds_vec = gpd.GeoDataFrame(data={"class": []},
                                              geometry=[], crs=crs)

            utils.write_geojson_to_gcp(cloudprob_path_dest, clouds_vec)

        # Copy floodmap shapefiles
        if not fsdest.exists(floodmap_path_dest) or overwrite:
            if pbar is not None:
                pbar.write(f"Saving floodmap {file_name}...")

            utils.write_geojson_to_gcp(floodmap_path_dest, floodmap)                
        
        # Copy S2 image
        if not fsdest.exists(s2_image_path_dest) or overwrite:
            if pbar is not None:
                pbar.write(f"Saving S2 image {file_name}...")
            
            _copy(s2_image_path, s2_image_path_dest, fs)

        # Copy permanent water
        if (permanent_water_image_path_dest is not None) and (not fsdest.exists(permanent_water_image_path_dest) or overwrite):
            if pbar is not None:
                pbar.write(f"Saving permanent water image {file_name}...")

            _copy(permanent_water_path, permanent_water_image_path_dest, fs)

    except Exception:
        warnings.warn(f"File input: {main_path} output S2 file: {s2_image_path_dest} problem when computing Ground truth")
        traceback.print_exc(file=sys.stdout)

        if not delete_if_error:
            return False

        fsdest = utils.get_filesystem(s2_image_path_dest)
        files_to_delete = [cloudprob_path_dest, gt_path_dest, meta_json_path_dest, permanent_water_image_path_dest,
                           s2_image_path_dest, floodmap_path_dest]
        for f in files_to_delete:
            if f and fsdest.exists(f):
                print(f"Deleting file {f}")
                fsdest.delete(f)

        return False

    return True


def _copy(file_or, file_dest, fsor):    
    if file_dest.startswith("gs"):
        fsor.copy(file_or, file_dest)
    else:
        fsor.get_file(file_or, file_dest)
    

def assert_element_consistency(output_path:Path, tiff_file_name:str, warn_permanent_water:bool=True):
    """
    Assert all elements in worldlfooods_output_files exists for the given tiff_file_name.

    Args:
        output_path: path to check e.g. Path("ml4cc_data_lake/0_DEV/2_Mart/worldfloods_v2_0/train")
        tiff_file_name: e.g. "SP6_20170502_WaterExtent_WetSoil_VillaRiva.tif"
        warn_permanent_water: warn if the permanent water layer is not found

    """
    cloudprob_path_dest, floodmap_path_dest, gt_path, meta_parent_path, permanent_water_image_path_dest, s2_image_path_dest = worldfloods_output_files(output_path, tiff_file_name)

    for p in [cloudprob_path_dest, floodmap_path_dest, gt_path, meta_parent_path, s2_image_path_dest]:
        assert p.check_if_file_exists(), f"{p.full_path} not found"

    if warn_permanent_water and not permanent_water_image_path_dest.check_if_file_exists():
        warnings.warn(f"{permanent_water_image_path_dest.full_path} not found")


def worldfloods_output_files(output_path:str, file_name:str,
                             permanent_water_available:bool=True,
                             clouds_available:bool=True,
                             mkdirs:bool=False) -> Tuple[str, str, str, str, Optional[str], str]:
    """
    For a given file (`tiff_file_name`) it returns the set of paths that the function generate_item produce.

    These paths are:
    - cloudprob_path_dest (.tif)
    - floodmap_path_dest. (.geojson)
    - gt_path (.tif)
    - meta_parent_path (.tif)
    - permanent_water_image_path_dest (.tif) or None if not permanent_water_available
    - s2_image_path_dest (.tif)

    Args:
        output_path: Path to produce the outputs
        file_name:
        permanent_water_available:
        clouds_available:
        mkdirs: make dirs if needed for the output paths

    Returns:
        cloudprob_path_dest, floodmap_path_dest, gt_path, meta_parent_path, permanent_water_image_path_dest, s2_image_path_dest

    """
    if permanent_water_available:
        permanent_water_image_path_dest = os.path.join(output_path, "PERMANENTWATERJRC", file_name+".tif").replace("\\", "/")
    else:
        permanent_water_image_path_dest = None

    output_path = str(output_path)
    s2_image_path_dest = os.path.join(output_path,"S2",file_name+".tif").replace("\\", "/")
    meta_parent_path = os.path.join(output_path,"meta",file_name+".json").replace("\\", "/")

    if clouds_available:
        cloudprob_path_dest = os.path.join(output_path, "cloud_vec",file_name+".geojson").replace("\\", "/")
    else:
        cloudprob_path_dest = None

    floodmap_path_dest = os.path.join(output_path,"floodmaps",file_name+".geojson").replace("\\", "/")
    gt_path = os.path.join(output_path,"gt",file_name+".tif").replace("\\", "/")

    # makedir if not gs
    if mkdirs and not s2_image_path_dest.startswith("gs"):
        fs = utils.get_filesystem(s2_image_path_dest)
        for f in [s2_image_path_dest, meta_parent_path, cloudprob_path_dest, floodmap_path_dest, gt_path, permanent_water_image_path_dest]:
            if f is not None:
                fs.makedirs(os.path.dirname(f), exist_ok=True)

    return cloudprob_path_dest, floodmap_path_dest, gt_path, meta_parent_path, permanent_water_image_path_dest, s2_image_path_dest
