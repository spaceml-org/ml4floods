import sys
import warnings
import traceback

import tqdm
import json
from ml4floods.data.utils import save_file_to_bucket

from ml4floods.data.io import save_groundtruth_tiff_rasterio
from ml4floods.data import utils
from ml4floods.data.ee_download import process_s2metadata
import os
from ml4floods.data.utils import GCPPath, read_json_from_gcp

from pathlib import Path
from typing import Optional, Callable, Tuple, Dict
import geopandas as gpd
import rasterio
import fsspec
from ml4floods.data import save_cog


CLOUDPROB_PARENT_PATH = "worldfloods/tiffimages"
PERMANENT_WATER_PARENT_PATH = "worldfloods/tiffimages/PERMANENTWATERJRC"
META_FLOODMAP_PARENT_PATH = "worldfloods/tiffimages/meta"
WORLDFLOODS_V0_BUCKET = "ml4floods"


def worldfloods_extra_gcp_paths(main_path: GCPPath) -> Tuple[gpd.GeoDataFrame, Optional[GCPPath], Optional[GCPPath], Dict, GCPPath]:
    """
    Given a pickle file in "gs://ml4cc_data_lake/{prod_dev}/1_Staging/WorldFloods it returns the rest of the files to
    create the full worldfloods registry (the corresponding floodmap, cloud probability and permanent water)

    Args:
        main_path: path to .piclke file in bucket

    Returns:
        GCPPaths with locations of corresponding  floodmap, cloud probability,  permanent water, meta_floodmap and sentinel2 image


    """
    assert main_path.check_if_file_exists(), f"File {main_path} does not exists"

    meta_floodmap = utils.read_pickle_from_gcp(main_path.full_path)

    # path to floodmap path
    floodmap_path = main_path.replace("/flood_meta/", "/floodmap/")
    floodmap_path = floodmap_path.replace(".pickle", ".geojson")

    assert floodmap_path.check_if_file_exists(), f"Floodmap not found in {floodmap_path}"

    path_aoi = os.path.dirname(os.path.dirname(floodmap_path.full_path))

    # open floodmap with geopandas
    floodmap = gpd.read_file(floodmap_path.full_path)
    floodmap_date = meta_floodmap['satellite date']

    # create permenant water path
    permanent_water_path = GCPPath(os.path.join(path_aoi, "PERMANENTWATERJRC", f"{floodmap_date.year}.tif"))

    if not permanent_water_path.check_if_file_exists():
        warnings.warn(f"Permanent water {permanent_water_path}. Will not be used")
        permanent_water_path = None

    csv_path = os.path.join(path_aoi, "S2", "s2info.csv")
    metadatas2 = process_s2metadata(csv_path)
    metadatas2 = metadatas2.set_index("names2file")

    assert any(metadatas2.s2available), f"Not available S2 files for {main_path}. {metadatas2}"

    # find corresponding s2_image_path for this image
    index = None
    s2_date = None
    for tup in metadatas2[metadatas2.s2available].itertuples():
        date_img = tup.datetime
        if (floodmap_date < date_img) or ((floodmap_date-date_img).total_seconds() / 3600. < 10):
            if s2_date is None:
                s2_date = date_img
                index = tup.Index
            else:
                if s2_date > date_img:
                    s2_date = date_img
                    index = tup.Index

    assert s2_date is not None, f"Not found valid S2 files for {main_path}. {metadatas2}"

    assert (s2_date - floodmap_date).total_seconds() / (3600. * 24) < 10, \
        f"Difference between S2 date {s2_date} and floodmap date {floodmap_date} is larger than 10 days"

    # add exact metadata from the csv file
    meta_floodmap["s2_date"] = s2_date
    meta_floodmap["names2file"] = index
    meta_floodmap["cloud_probability"] = metadatas2.loc[index, "cloud_probability"]
    meta_floodmap["valids"] = metadatas2.loc[index, "valids"]

    s2_image_path = GCPPath(os.path.join(path_aoi, "S2", index+".tif"))

    return floodmap, None, permanent_water_path, meta_floodmap, s2_image_path


def worldfloods_old_gcp_paths(main_path: GCPPath) -> Tuple[gpd.GeoDataFrame, GCPPath, Optional[GCPPath], Dict, GCPPath]:
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

    meta_floodmap = read_json_from_gcp(meta_floodmap_path.full_path)

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
        file_name:
        pbar: optional progress bar with method description.
        gt_fun: one of ml4floods.data.create_gt.generate_land_water_cloud_gt or ml4floods.data.create_gt.generate_water_cloud_binary_gt.
        This function determines how the ground truth is created from the input products.
        delete_if_error: whether to delete the generated files if an error is risen
        paths_function: function to get the paths of the files for the given main_path

    Returns:
        True if success in creating all the products

    """
    name_path_local = gt_fun.__name__.replace(".", "-") + paths_function.__name__.replace(".", "-")
    local_path = Path(".").joinpath(name_path_local)
    os.makedirs(local_path, exist_ok=True)

    main_path = GCPPath(main_path)
    name = os.path.splitext(os.path.basename(main_path.file_name))[0]
    fs = fsspec.filesystem("gs")

    try:
        # Get input files and check that they all exist
        floodmap, cloudprob_path, permanent_water_path, metadata_floodmap, s2_image_path = paths_function(main_path)

        # get output files
        cloudprob_path_dest, floodmap_path_dest, gt_path_dest, meta_json_path_dest, permanent_water_image_path_dest, s2_image_path_dest = worldfloods_output_files(
            output_path, file_name, permanent_water_path is not None)
    except Exception:
        warnings.warn(f"File {main_path.file_name} problem when computing input/output names")
        traceback.print_exc(file=sys.stdout)
        return False
    try:
        # generate gt, gt meta and copy all files to path_write
        if not gt_path_dest.check_if_file_exists() or not meta_json_path_dest.check_if_file_exists() or overwrite:
            if pbar is not None:
                pbar.set_description(f"Generating Ground Truth {name}...")

            with rasterio.open(s2_image_path.full_path) as rst:
                bands = rst.descriptions
                cloudprob_in_lastband = (len(bands) > 14) and (bands[14] == "probability")

            gt, gt_meta = gt_fun(
                s2_image_path.full_path,
                floodmap,
                metadata_floodmap=metadata_floodmap,
                keep_streams=True,
                cloudprob_image_path=cloudprob_path if cloudprob_path is None else cloudprob_path.full_path,
                cloudprob_in_lastband=cloudprob_in_lastband,
                permanent_water_image_path=permanent_water_path if permanent_water_path is None else permanent_water_path.full_path,  # Could be None!
            )

            if len(gt.shape) == 2:
                gt = gt[None]

            if pbar is not None:
                pbar.set_description(f"Saving GT {name}...")

            save_cog.save_cog(gt, gt_path_dest.full_path,
                              {"crs": gt_meta["crs"], "transform":gt_meta["transform"] ,"RESAMPLING": "NEAREST",
                               "compression": "lzw", "nodata": 0}, # In both gts 0 is nodata
                              tags=gt_meta)

            # upload meta json to bucket
            if pbar is not None:
                pbar.set_description(f"Saving meta {name}...")

            # save meta in local json file
            gt_meta["crs"] = str(gt_meta["crs"])
            gt_meta["transform"] = [gt_meta["transform"].a, gt_meta["transform"].b, gt_meta["transform"].c,
                                    gt_meta["transform"].d, gt_meta["transform"].e, gt_meta["transform"].f]

            utils.write_json_to_gcp(meta_json_path_dest.full_path, gt_meta)

        # Copy floodmap shapefiles
        if not floodmap_path_dest.check_if_file_exists() or overwrite:
            if pbar is not None:
                pbar.set_description(f"Saving floodmap {name}...")

            utils.write_geojson_to_gcp(floodmap_path_dest.full_path, floodmap)

        # Copy cloudprob, S2 and permanent water
        if cloudprob_path is not None and (not cloudprob_path_dest.check_if_file_exists() or overwrite):
            if pbar is not None:
                pbar.set_description(f"Saving cloud probs {name}...")
            fs.copy(cloudprob_path.full_path, cloudprob_path_dest.full_path)

        if not s2_image_path_dest.check_if_file_exists() or overwrite:
            if pbar is not None:
                pbar.set_description(f"Saving S2 image {name}...")

            fs.copy(s2_image_path.full_path, s2_image_path_dest.full_path)

        if (permanent_water_image_path_dest is not None) and (not permanent_water_image_path_dest.check_if_file_exists() or overwrite):
            if pbar is not None:
                pbar.set_description(f"Saving permanent water image {name}...")

            fs.copy(permanent_water_path.full_path, permanent_water_image_path_dest.full_path)

    except Exception:
        warnings.warn(f"File input: {main_path.file_name} output S2 file: {s2_image_path_dest.full_path} problem when computing Ground truth")
        traceback.print_exc(file=sys.stdout)

        if not delete_if_error:
            return False

        files_to_delete = [cloudprob_path_dest, gt_path_dest, meta_json_path_dest, permanent_water_image_path_dest,
                           s2_image_path_dest]
        if floodmap_path_dest.suffix == "shp":
            files_to_delete.extend([GCPPath(f) for f in floodmap_path_dest.get_files_in_parent_directory_with_name()])
        else:
            files_to_delete.append(floodmap_path_dest)
        for f in files_to_delete:
            if f.check_if_file_exists():
                print(f"Deleting file {f.full_path}")
                f.delete()

        return False

    return True


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
                             permanent_water_available:bool=True) -> Tuple[GCPPath, GCPPath, GCPPath, GCPPath, Optional[GCPPath], GCPPath]:
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

    Returns:
        cloudprob_path_dest, floodmap_path_dest, gt_path, meta_parent_path, permanent_water_image_path_dest, s2_image_path_dest

    """
    if permanent_water_available:
        permanent_water_image_path_dest = GCPPath(os.path.join(output_path, "PERMANENTWATERJRC", file_name+".tif"))
    else:
        permanent_water_image_path_dest = None

    output_path = str(output_path)
    s2_image_path_dest = GCPPath(os.path.join(output_path,"S2",file_name+".tif"))
    meta_parent_path = GCPPath(os.path.join(output_path,"meta",file_name+".json"))
    cloudprob_path_dest = GCPPath(os.path.join(output_path, "cloudprob",file_name+".tif"))
    floodmap_path_dest = GCPPath(os.path.join(output_path,"floodmaps",file_name+".geojson"))
    gt_path = GCPPath(os.path.join(output_path,"gt",file_name+".tif"))

    return cloudprob_path_dest, floodmap_path_dest, gt_path, meta_parent_path, permanent_water_image_path_dest, s2_image_path_dest
