import sys
import warnings
import traceback

import tqdm
import json
from ml4floods.data.utils import save_file_to_bucket

from ml4floods.data.io import save_groundtruth_tiff_rasterio
import os
from ml4floods.data.utils import GCPPath, CustomJSONEncoder, read_json_from_gcp

from pathlib import Path
from typing import Union, Optional, Callable, Tuple, Dict
import geopandas as gpd

CLOUDPROB_PARENT_PATH = "worldfloods/tiffimages"
PERMANENT_WATER_PARENT_PATH = "worldfloods/tiffimages/PERMANENTWATERJRC"
META_FLOODMAP_PARENT_PATH = "worldfloods/tiffimages/meta"
WORLDFLOODS_V0_BUCKET = "ml4floods"


def worldfloods_old_gcp_paths(s2_image_path: GCPPath) -> Tuple[gpd.GeoDataFrame, GCPPath, Optional[GCPPath], Dict]:
    """
    Given a S2 tiff file in the V0 WorldFloods dataset it returns the rest of the anciliary files to
    the corresponding floodmap, cloud probability and permanent water

    Args:
        s2_image_path: S2 path in gs://ml4floods/worldfloods/public folder

    Returns:
        GCPPaths with locations of corresponding  floodmap, cloud probability and permanent water

    Examples
        >>> s2_path_file = "gs://ml4floods/worldfloods/public/train/S2/EMSR260_09RUBIERA_GRA_v2_observed_event_a.tif"
        >>> worldfloods_old_gcp_paths(GCPPath(s2_path_file))

    (GCPPath(full_path='gs://ml4floods/worldfloods/public/train/floodmaps/EMSR260_09RUBIERA_GRA_v2_observed_event_a.shp', bucket_id='ml4floods', parent_path='worldfloods/public/train/floodmaps', file_name='EMSR260_09RUBIERA_GRA_v2_observed_event_a.shp', suffix='shp'),
     GCPPath(full_path='gs://ml4floods/worldfloods/tiffimages/cloudprob/EMSR260_09RUBIERA_GRA_v2_observed_event_a.tif', bucket_id='ml4floods', parent_path='worldfloods/tiffimages/cloudprob', file_name='EMSR260_09RUBIERA_GRA_v2_observed_event_a.tif', suffix='tif'),
     'gs://ml4floods/worldfloods/tiffimages/PERMANENTWATERJRC/EMSR260_09RUBIERA_GRA_v2_observed_event_a.tif')

    """
    assert s2_image_path.check_if_file_exists(), f"Clouds not found in {s2_image_path}"

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

    return floodmap, cloudprob_path, permanent_water_path, meta_floodmap


def generate_item(s2_image_path:str, output_path:Union[str, Path],
                  overwrite:bool=False, pbar:Optional[tqdm.tqdm]=None, gt_fun:Callable=None, delete_if_error:bool=True) -> bool:
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
        s2_image_path: Path to anciliary S2 image. The process will search for other relevant information to create all the
        aforementioned products.
        output_path: Folder where the item will be written. See fun worldfloods_output_files for corresponding output file naming convention.
        overwrite: if False it will not overwrite existing products in path_write folder.
        pbar: optional progress bar with method description.
        gt_fun: one of ml4floods.data.create_gt.generate_land_water_cloud_gt or ml4floods.data.create_gt.generate_water_cloud_binary_gt.
        This function determines how the ground truth is created from the input products.
        delete_if_error: whether to delete the generated files if an error is risen

    Returns:
        True if success in creating all the products

    """

    local_path = Path(".").joinpath("cache_datasets")
    os.makedirs(local_path, exist_ok=True)

    s2_image_path = GCPPath(s2_image_path)
    name = os.path.splitext(os.path.basename(s2_image_path.file_name))[0]

    try:
        # Get input files and check that they all exist
        floodmap, cloudprob_path, permanent_water_path, metadata_floodmap = worldfloods_old_gcp_paths(
            s2_image_path)

        # get output files
        cloudprob_path_dest, floodmap_path_dest, gt_path_dest, meta_json_path_dest, permanent_water_image_path_dest, s2_image_path_dest = worldfloods_output_files(
            output_path, s2_image_path.file_name, permanent_water_path is not None)
    except:
        warnings.warn(f"File {s2_image_path} problem when computing input/output names")
        traceback.print_exc(file=sys.stdout)
        return False
    try:
        # generate gt, gt meta and copy all files to path_write
        if not gt_path_dest.check_if_file_exists() or not meta_json_path_dest.check_if_file_exists() or overwrite:
            if pbar is not None:
                pbar.set_description(f"Generating Ground Truth {name}...")

            gt, gt_meta = gt_fun(
                s2_image_path.full_path,
                floodmap,
                metadata_floodmap=metadata_floodmap,
                keep_streams=True,
                cloudprob_image_path=cloudprob_path.full_path,
                permanent_water_image_path=permanent_water_path if permanent_water_path is None else permanent_water_path.full_path,  # Could be None!
            )

            # save ground truth in local file
            gt_local_path = local_path.joinpath(gt_path_dest.file_name)
            save_groundtruth_tiff_rasterio(
                gt,
                str(gt_local_path),
                gt_meta=gt_meta,
                crs=gt_meta["crs"],
                transform=gt_meta["transform"],
            )

            # save meta in local json file
            meta_local_file = str(local_path.joinpath(s2_image_path.file_name)).replace(".tif", ".json")
            del gt_meta["crs"]
            del gt_meta["transform"]
            with open(meta_local_file, "w") as fh:
                json.dump(gt_meta, fh, cls=CustomJSONEncoder)

            # upload ground truth to bucket
            if pbar is not None:
                pbar.set_description(f"Saving GT {name}...")
            save_file_to_bucket(
                gt_path_dest.full_path, str(local_path.joinpath(gt_path_dest.file_name))
            )
            # delete local file
            gt_local_path.unlink()

            # upload meta json to bucket
            if pbar is not None:
                pbar.set_description(f"Saving meta {name}...")
            save_file_to_bucket(
                meta_json_path_dest.full_path, meta_local_file
            )
            Path(meta_local_file).unlink()

        # Copy floodmap shapefiles
        if pbar is not None:
            pbar.set_description(f"Saving floodmap {name}...")

        floodmap_local_file = str(local_path.joinpath(s2_image_path.file_name)).replace(".tif", ".geojson")
        floodmap.to_file(floodmap_local_file, driver="GeoJSON")
        save_file_to_bucket(floodmap_path_dest.full_path, floodmap_local_file)

        # Copy cloudprob, S2 and permanent water
        if not cloudprob_path_dest.check_if_file_exists() or overwrite:
            if pbar is not None:
                pbar.set_description(f"Saving cloud probs {name}...")
            cloudprob_path.transfer_file_to_bucket_gsutils(
                cloudprob_path_dest.full_path, file_name=True
            )

        if not s2_image_path_dest.check_if_file_exists() or overwrite:
            if pbar is not None:
                pbar.set_description(f"Saving S2 image {name}...")
            s2_image_path.transfer_file_to_bucket_gsutils(
                s2_image_path_dest.full_path, file_name=True
            )
        if permanent_water_image_path_dest is not None and not permanent_water_image_path_dest.check_if_file_exists() or overwrite:
            if pbar is not None:
                pbar.set_description(f"Saving permanent water image {name}...")
            permanent_water_path.transfer_file_to_bucket_gsutils(
                permanent_water_image_path_dest.full_path, file_name=True
            )
    except:
        warnings.warn(f"File {s2_image_path} problem when computing Ground truth")
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


def worldfloods_output_files(output_path:Path, tiff_file_name:str, permanent_water_available:bool=True) -> Tuple[GCPPath, GCPPath, GCPPath, GCPPath, Optional[GCPPath], GCPPath]:
    """
    For a given file (`tiff_file_name`) it returns the set of paths that the function generate_item produce.

    These paths are:
    - cloudprob_path_dest (.tif)
    - floodmap_path_dest. Folder with shp files
    - gt_path (.tif)
    - meta_parent_path (.tif)
    - permanent_water_image_path_dest (.tif) or None if not permanent_water_available
    - s2_image_path_dest (.tif)

    Args:
        output_path: Path to produce the outputs
        tiff_file_name:
        permanent_water_available:

    Returns:
        cloudprob_path_dest, floodmap_path_dest, gt_path, meta_parent_path, permanent_water_image_path_dest, s2_image_path_dest

    """
    if permanent_water_available:
        permanent_water_image_path_dest = GCPPath(
            str(
                output_path
                    .joinpath("PERMANENTWATERJRC")
                    .joinpath(tiff_file_name)
            )
        )
    else:
        permanent_water_image_path_dest = None
    s2_image_path_dest = GCPPath(
        str(
            output_path
                .joinpath("S2")
                .joinpath(tiff_file_name)
        )
    )
    meta_parent_path = GCPPath(str(
        output_path
            .joinpath("meta")
            .joinpath(tiff_file_name.replace(".tif", ".json"))
    ))
    cloudprob_path_dest = GCPPath(
        str(
            output_path
                .joinpath("cloudprob")
                .joinpath(tiff_file_name)
        )
    )

    floodmap_path_dest = GCPPath(
        str(
            output_path
                .joinpath("floodmaps")
                .joinpath(tiff_file_name.replace(".tif", ".gejson"))
        )
    )

    # replace parent path
    gt_path = GCPPath(str(output_path
                          .joinpath("gt")
                          .joinpath(tiff_file_name)))

    return cloudprob_path_dest, floodmap_path_dest, gt_path, meta_parent_path, permanent_water_image_path_dest, s2_image_path_dest