"""
This script contains all the utility functions that are not specific to a particular kind of dataset.
These are mainly used for explorations, testing, and demonstrations.
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import warnings

import geopandas as gpd
import numpy as np
from google.cloud import storage
from shapely.geometry import Polygon, mapping
from shapely.ops import cascaded_union
from datetime import datetime

from ml4floods.data.config import CLASS_LAND_COPERNICUSEMSHYDRO
from dataclasses import dataclass, field
import subprocess
import pickle
import fsspec
from contextlib import contextmanager
import rasterio


@dataclass
class GCPPath:
    full_path: str
    bucket_id: str = field(default=None)
    parent_path: str = field(default=None)
    file_name: str = field(default=None)
    suffix: str = field(default=None)

    def __init__(self, full_path):

        # trick to ensure the prefix is always there
        full_path = add_gcp_prefix(remove_gcp_prefix(full_path))

        self.full_path = full_path
        #         print(self.full_path)
        self.bucket_id = str(Path(full_path.split("gs://")[1]).parts[0])

        #         print(self.bucket_id)
        self.parent_path = str(Path(full_path.split(self.bucket_id)[1]).parent)[1:]
        #         print(self.parent_path)
        self.file_name = str(Path(full_path).name)
        #         print(self.file_name)
        self.suffix = self.file_name.split(".")[1]

    #         print(self.suffix)

    def get_files_in_parent_directory(self, **kwargs):
        # initialize client
        client = storage.Client(**kwargs)
        # get bucket
        bucket = client.get_bucket(self.bucket_id)
        # get blob

        blobs = bucket.list_blobs(prefix=self.parent_path)
        # check if it exists

        files = ["gs://" + str(Path(self.bucket_id).joinpath(x.name)) for x in blobs]
        return files

    def get_files_in_parent_directory_with_suffix(self, suffix:str, **kwargs):
        # initialize client
        client = storage.Client(**kwargs)
        # get bucket
        bucket = client.get_bucket(self.bucket_id)
        # get blob

        blobs = bucket.list_blobs(prefix=self.parent_path)
        # check if it exists
        files = [
            "gs://" + str(Path(self.bucket_id).joinpath(x.name))
            for x in blobs
            if str(Path(x.name).suffix) == suffix
        ]
        return files

    def get_files_in_parent_directory_with_name(
        self, name: Optional[str] = None, **kwargs
    ):
        # initialize client
        client = storage.Client(**kwargs)
        # get bucket
        bucket = client.get_bucket(self.bucket_id)
        # get blob

        blobs = bucket.list_blobs(prefix=self.parent_path)

        if name is None:
            name = self.get_file_name_stem()
        # check if it exists
        files = [
            "gs://" + str(Path(self.bucket_id).joinpath(x.name))
            for x in blobs
            if name in str(Path(x.name))
        ]
        return files

    def get_file_name_stem(self):
        return str(Path(self.file_name).stem)

    def transfer_file_to_bucket(
        self, destination_bucket_name: str, destination_file_path: str, **kwargs
    ):
        """Transfers using the google-storage package

        Args:
            destination_file_path (str): the destination to the file path
            name (bool): flag to check for the name or not

        Returns:
            None

        Examples:
            Example with no name attached

            >>> origin = 'gs://bucket/path/to/my/file.suffix'
            >>> my_path_class = GCPPath(origin)
            >>> dest = 'gs://newbucket/new/path/'
            >>> my_path_class.transfer_file_to_bucket(dest)

            Another example, except we have the name attached:

            >>> origin = 'gs://bucket/path/to/my/file.suffix'
            >>> my_path_class = GCPPath(origin)
            >>> dest = 'gs://newbucket/new/path/to/my/file.suffix'
            >>> my_path_class.transfer_file_to_bucket(dest, name=True)
        """

        storage_client = storage.Client(**kwargs)
        source_bucket = storage_client.get_bucket(self.bucket_id)
        source_blob = source_bucket.blob(self.get_file_path())

        destination_bucket = storage_client.get_bucket(destination_bucket_name)

        destination_blob_name = str(
            Path(destination_file_path).joinpath(self.file_name)
        )
        # copy to new destination
        new_blob = source_bucket.copy_blob(
            source_blob, destination_bucket, destination_blob_name
        )

        return self

    def transfer_file_to_bucket_gsutils(
        self, destination_file_path: str, file_name: bool = False, **kwargs
    ):
        """Transfers using the gsutils package
        Very useful function when we have files that are quite large.
        The standard google-storage package doesn't work well with these
        types of files.

        Args:
            destination_file_path (str): the destination to the file path

        Returns:
            None

        Examples:
            Example with no name attached

            >>> origin = 'gs://bucket/path/to/my/file.suffix'
            >>> my_path_class = GCPPath(origin)
            >>> dest = 'gs://newbucket/new/path/'
            >>> my_path_class.transfer_file_to_bucket_gsutils(dest)

            Another example, except we have the name attached:

            >>> origin = 'gs://bucket/path/to/my/file.suffix'
            >>> my_path_class = GCPPath(origin)
            >>> dest = 'gs://newbucket/new/path/to/my/file.suffix'
            >>> my_path_class.transfer_file_to_bucket_gsutils(dest, name=True)
        """

        # remove prefix
        destination_file_path = remove_gcp_prefix(destination_file_path)
        # join paths
        if not file_name:
            destination_file_path = str(
                Path(destination_file_path).joinpath(self.file_name)
            )
        # add prefix
        destination_file_path = add_gcp_prefix(destination_file_path)

        subprocess.call(
            ["gsutil", "cp", f"{self.full_path}", f"{destination_file_path}"]
        )

        return self

    def download_file_from_bucket(self, destination_path: str):

        client = storage.Client()

        bucket = client.get_bucket(self.bucket_id)
        # get blob
        blob = bucket.get_blob(self.get_file_path())

        # create directory if needed
        create_folder(destination_path)

        # get full path
        destination_file_name = Path(destination_path).joinpath(self.file_name)

        # download data
        blob.download_to_filename(str(destination_file_name))

        return destination_file_name

    def check_if_file_exists(self, **kwargs):
        # initialize client
        client = storage.Client(**kwargs)
        # get bucket
        bucket = client.get_bucket(self.bucket_id)
        # get blob
        blob = bucket.blob(self.get_file_path())
        # check if it exists
        return blob.exists()

    def delete(self, **kwargs):
        # initialize client
        client = storage.Client(**kwargs)
        # get bucket
        bucket = client.get_bucket(self.bucket_id)
        # get blob
        blob = bucket.blob(self.get_file_path())
        blob.delete()

    #         return get_files_in_bucket_directory(self.bucket_id, directory=self.parent_path, suffix=self.suffix)

    def get_file_path(self):
        return str(Path(self.parent_path).joinpath(self.file_name))

    def replace_bucket(self, bucket_id):
        self.bucket_id = bucket_id
        return self

    def replace_file_name(self, file_name):
        self.file_name = file_name
        return self

    def replace(self, original: str, replacement: str):

        full_path = self.full_path.replace(original, replacement)

        #         self.__init__(full_path)

        return GCPPath(full_path)


def download_data_from_bucket(
    filenames: List[str],
    destination_dir: str,
    # ml_split: str = "train",
    bucket_id: Optional[str] = None,
) -> None:
    """Function to download data from the bucket to a local destination directory.
    This function differs from the save_file_from_bucket() function in that
    it takes as input a list of filenames to be downloaded compared to save_file_from_bucket()
    which deals with only a single file.
    Wraps around the save_file_from_bucket() function to download the list of files.

    Args:
        filenames (List[str]): List of filenames to be downloaded from the bucket.
        destination_dir (str): Path of the destination directory.
        bucket_id (str, optional): Name of the bucket being used to download the files.
            Defaults to None.
    """

    for ifile in filenames:
        bucket_id = str(Path(ifile).parts[0])

        file_name = str(Path(*Path(ifile).parts[1:]))
        save_file_from_bucket(
            bucket_id, file_name=file_name, destination_file_path=destination_dir
        )


def check_file_in_bucket_exists_gs(gs_path: str, **kwargs) -> bool:
    """
    Function to check if the file in the bucket exist utilizing Google Cloud Storage
    (GCP) blobs. Same as the check_file_in_bucket_exists() function but it takes as
    input the complete gcp gs path.

    Args:
      bucket_name (str): a string corresponding to the name of the GCP bucket.
      filename_full_path (str): a string containing the full path from bucket to file.

    Returns:
      A boolean value corresponding to the existence of the file in the bucket.
    """
    # initialize client
    client = storage.Client(**kwargs)

    # get bucket
    bucket_id = gs_path.split("gs://")[-1].split("/")[0]
    bucket = client.get_bucket(bucket_id)

    # get blob
    filename_full_path = gs_path.replace(f"gs://{bucket_id}/", "")
    blob = bucket.blob(filename_full_path)

    # check if it exists
    return blob.exists()


def check_file_in_bucket_exists(
    bucket_name: str, filename_full_path: str, **kwargs
) -> bool:
    """
    Function to check if the file in the bucket exist utilizing Google Cloud Storage
    (GCP) blobs.

    Args:
      bucket_name (str): a string corresponding to the name of the GCP bucket.
      filename_full_path (str): a string containing the full path from bucket to file.

    Returns:
      A boolean value corresponding to the existence of the file in the bucket.
    """
    # initialize client
    client = storage.Client(**kwargs)
    # get bucket
    bucket = client.get_bucket(bucket_name)
    # get blob
    blob = bucket.blob(filename_full_path)
    # check if it exists
    return blob.exists()


def load_json_from_bucket(bucket_name: str, filename: str, **kwargs) -> Dict:
    """
    Function to load the json data for the WorldFloods bucket using the filename
    corresponding to the image file name. The filename corresponds to the full
    path following the bucket name through intermediate directories to the final
    json file name.

    Args:
      bucket_name (str): the name of the Google Cloud Storage (GCP) bucket.
      filename (str): the full path following the bucket_name to the json file.

    Returns:
      The unpacked json data formatted to a dictionary.
    """
    # initialize client
    client = storage.Client(**kwargs)
    # get bucket
    bucket = client.get_bucket(bucket_name)
    # get blob
    blob = bucket.blob(filename)
    # check if it exists
    # TODO: wrap this within a context
    return json.loads(blob.download_as_string(client=None))


# def generate_list_of_files(bucket_id: str, file_path: str):
#     """Generate a list of files within the mentioned filepath from the bucket."""

#     return None


def filter_land(gpddats: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """ Filter land from pandas dataframe (land class specified in hydrology maps from CopernicusEMS) """
    isnot_land = gpddats.obj_type.apply(
        lambda g: g not in CLASS_LAND_COPERNICUSEMSHYDRO
    )

    if np.sum(isnot_land) == gpddats.shape[0]:
        return gpddats

    gpddats_notland = gpddats[isnot_land].copy()
    if gpddats_notland.shape[0] == 0:
        return gpddats_notland

    land_geometries = gpddats.geometry[~isnot_land]
    land_geometries = cascaded_union(land_geometries.tolist())

    # Not all polygons are valid: filter to valid gpddats_notland
    gpddats_notland_valid = gpddats_notland[gpddats_notland["geometry"].is_valid]

    gpddats_notland_valid["geometry"] = gpddats_notland_valid.geometry.apply(
        lambda g: g.difference(land_geometries)
    )

    return gpddats_notland_valid


def filter_pols(gpddats: gpd.GeoDataFrame, pol_shapely: Polygon) -> gpd.GeoDataFrame:
    """ filter pols that do not intersects pol_shapely """
    gpddats_cp = gpddats[~(gpddats.geometry.isna() | gpddats.geometry.is_empty)].copy()

    return (
        gpddats_cp[gpddats_cp.geometry.apply(lambda g: g.intersects(pol_shapely))]
        .reset_index()
        .copy()
    )


def generate_list_of_files(bucket_id: str, file_path):
    """Generate a list of files from the bucket."""
    return None


def save_file_from_bucket(bucket_id: str, file_name: str, destination_file_path: str):
    """Function to save a file from a bucket to the mentioned destination file path.

    Args:
        bucket_id (str): the name of the bucket
        file_name (str): the name of the file in bucket (include the directory)
        destination_file_path (str): the directory of where you want to save the
            data locally (not including the filename)

    Returns:
        None: Returns nothing.

    Examples:
        >>> bucket_id = sample_bucket
        >>> file_name = 'path/to/file/and/file.csv'
        >>> dest = 'path/in/bucket/'
        >>> save_file_from_bucket(
        ...     bucket_id=bucket_id,
        ...     file_name=file_name,
        ...     destination_file_path=dest
        ... )
    """

    client = storage.Client()

    bucket = client.get_bucket(bucket_id)
    # get blob
    blob = bucket.get_blob(file_name)

    # create directory if needed
    create_folder(destination_file_path)

    # get full path
    destination_file_name = Path(destination_file_path).joinpath(
        file_name.split("/")[-1]
    )
    # download data
    blob.download_to_filename(str(destination_file_name))

    return None


def open_file_from_bucket(target_directory: str):
    """Function to open a file directly from the bucket.

    Args:
        target_directory (str): Complete filepath of the file to be opened
            within the session.

    Returns:
        google.cloud.storage.blob.Blob: Returns the blob of file
            that is read into memory within the current session.

    Example:
        >>> target_directory = 'path/to/file/and/file.pkl'
        >>> open_file_from_bucket(target_directory)
    """

    bucket_id, file_path, file_name = parse_gcp_path(target_directory)

    file_path = str(Path(file_path).joinpath(file_name))[1:]
    client = storage.Client()

    bucket = client.get_bucket(bucket_id)
    # get blob
    blob = bucket.get_blob(file_path)

    # download data
    blob = blob.download_as_string()

    return blob


def save_file_to_bucket(target_directory: str, source_directory: str):
    """Function to save file to a bucket.

    Args:
        target_directory (str): Destination file path.
        source_directory (str): Source file path

    Returns:
        None: Returns nothing.

    Examples:
        >>> target_directory = 'target/path/to/file/.pkl'
        >>> source_directory = 'source/path/to/file/.pkl'
        >>> save_file_to_bucket(target_directory)
    """

    client = storage.Client()

    bucket_id, _, _ = parse_gcp_path(target_directory)
    file_path = target_directory.split(bucket_id)[1][1:]

    bucket = client.get_bucket(bucket_id)

    # get blob
    blob = bucket.blob(file_path)

    # upload data
    blob.upload_from_filename(source_directory)

    return None


def check_path_exists(path: str) -> None:
    """Checks if the given exists.

    Args:
        path (str): Input file path

    Raises:
        ValueError: Raises an error in case the file path does not exist

    Returns:
        None: Returns nothing.
    """
    if not Path(path).is_dir():
        raise ValueError(f"Unrecognized path: {str(Path(path))}")
    return None


def create_folder(directory: str) -> None:
    """Function to create directory if it doesn't exist.

    Args:
        directory (str): directory to be created if it doesn't already exist.

    Example:
        Typical usage example:

        >>> from ml4floods.data.utils import create_folder
        >>> directory = "./temp"
        >>> create_folder(directory)
    """

    try:
        Path(directory).mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"Folder '{directory}' Is Already There.")
    else:
        print(f"Folder '{directory}' is created.")


def get_files_in_directory_gcp(bucket_id: str, directory: str, **kwargs) -> List[str]:
    """Function to return the list of files within a given directory.

    Args:
        directory (str): Directory path to get the file list from.
        suffix (str): file extension to be listed

    Returns:
        List[str]: Returns the list of files that match the given extension
            within the given directory.
    """
    client = storage.Client(**kwargs)
    # get bucket
    bucket = client.get_bucket(bucket_id)
    # get blob

    blobs = bucket.list_blobs(prefix=directory)
    # check if it exists

    # for
    files = [
        str(Path(bucket_id).joinpath(x.name))
        for x in blobs
        #     if str(Path(x.name).suffix) == suffix
    ]

    return files


def get_files_in_directory(directory: str, suffix: str) -> List[str]:
    """Function to return the list of files within a given directory.

    Args:
        directory (str): Directory path to get the file list from.
        suffix (str): file extension to be listed

    Returns:
        List[str]: Returns the list of files that match the given extension
            within the given directory.
    """
    p = Path(directory).glob(f"*{suffix}")
    files = [str(x) for x in p if x.is_file()]
    return files


# TODO: This is a redundant function.
# Refactor all the code to use only one of
# these two functions and get rid of the redundant function.
def get_filenames_in_directory(directory: str, suffix: str) -> List[str]:
    """Function to return the list of files within a given directory.

    Args:
        directory (str): Directory path to get the file list from.
        suffix (str): file extension to be listed

    Returns:
        List[str]: Returns the list of files that match the given extension
            within the given directory.
    """
    p = Path(directory).glob(f"*{suffix}")
    files = [str(x.name) for x in p if x.is_file()]
    return files


# def get_files_in_bucket_directory(
#     bucket_id: str, directory: str, suffix: str
# ) -> List[str]:
#     p = Path(directory).glob(f"*{suffix}")
#     files = [str(x) for x in p if x.is_file()]
#     return files


def get_files_in_bucket_directory_gs(
    gs_path: str, suffix: str = None, substring: str = None, **kwargs
) -> List[str]:
    """Function to return a list of files in gcp bucket path.
    This function is different from the `get_files_in_bucket_directory()` function
    because this function takes the entire gcp path as input instead of components of the path.
    Args:
        gs_path (str): the entire file/directory path stored in the gcp bucket
        suffix (str): the filename suffix, e.g. '.tif'
    Returns:
        files (List[str]): a list of filenames with the fullpaths
    """

    # initialize client
    client = storage.Client(**kwargs)

    # get bucket
    bucket_id = gs_path.split("gs://")[-1].split("/")[0]
    bucket = client.get_bucket(bucket_id)

    # get blob
    directory = gs_path.replace(f"gs://{bucket_id}/", "")
    blobs = bucket.list_blobs(prefix=directory)

    # check if it exists
    files = [
        f"gs://{x.bucket.name}/{str(x.name)}"
        for x in blobs
        if ((suffix is None) or (str(Path(x.name).suffix) == suffix))
    ]
    if substring is not None:
        files = [f for f in files if substring in os.path.basename(f)]

    return files


def get_files_in_bucket_directory(
    bucket_id: str, directory: str, suffix: str = None, **kwargs
) -> List[str]:
    """Function to return a list of files in bucket directory
    Args:
        bucket_id (str): the bucket name to query
        directory (str): the directory within the bucket to query
        suffix (str): the filename suffix, e.g. '.tif'
        full_path (bool): whether to add the full path to filenames or not
    Returns:
        files (List[str]): a list of filenames with the fullpaths
    """

    # initialize client
    client = storage.Client(**kwargs)
    # get bucket
    bucket = client.get_bucket(bucket_id)
    # get blob
    blobs = bucket.list_blobs(prefix=directory)
    # check if it exists

    files = [
        str(x.name)
        for x in blobs
        if ((suffix is None) or (str(Path(x.name).suffix) == suffix))
    ]
    return files


class CustomJSONEncoder(json.JSONEncoder):

    def default(self, obj_to_encode):
        """Pandas and Numpy have some specific types that we want to ensure
        are coerced to Python types, for JSON generation purposes. This attempts
        to do so where applicable.
        """
        # Pandas dataframes have a to_json() method, so we'll check for that and
        # return it if so.
        if hasattr(obj_to_encode, "to_json"):
            return obj_to_encode.to_json()
        # Numpy objects report themselves oddly in error logs, but this generic
        # type mostly captures what we're after.
        if isinstance(obj_to_encode, np.generic):
            return obj_to_encode.item()
        # ndarray -> list, pretty straightforward.
        if isinstance(obj_to_encode, np.ndarray):
            return obj_to_encode.tolist()
        if isinstance(obj_to_encode, Polygon):
            return mapping(obj_to_encode)
        if isinstance(obj_to_encode, pd.Timestamp):
            return obj_to_encode.isoformat()
        if isinstance(obj_to_encode, datetime):
            return obj_to_encode.isoformat()
        # torch or tensorflow -> list, pretty straightforward.
        if hasattr(obj_to_encode, "numpy"):
            return obj_to_encode.numpy().tolist()
        # If none of the above apply, we'll default back to the standard JSON encoding
        # routines and let it work normally.
        return super().default(obj_to_encode)


def parse_gcp_path(full_path) -> Tuple[str]:
    """Parse the bucket"""
    # parse the components
    bucket_id = str(Path(full_path.split("gs://")[1]).parts[0])
    file_path = str(Path(full_path.split(bucket_id)[1]).parent)
    file_name = str(Path(full_path).name)

    return bucket_id, file_path, file_name


def copy_file_between_gcpbuckets(
    source_bucket_name: str,
    source_file_path: str,
    destination_bucket_name: str,
    destination_blob_name: str,
    **kwargs,
) -> None:
    """
    Function for copying files between directories or buckets. it will use GCP's copy
    function.

    Args:
        source_bucket_name (str): name of SOURCE bucket
        source_file_path (str): name of SOURCE file path (without bucket name)
        destination_bucket_name (str): name of DESTINATION bucket
        destination_blob_name (str): name of DESTINATION file path (without bucket name)

    Examples:
        >>> source_bucket_name = "bucket_id_source"
        >>> destination_bucket_name = "destination_bucket_name"
        >>> file_path_from = "path/to/data.tif"
        >>> file_path_to = "path/to/data.tif"
        >>> mv_blob(source_bucket_name, file_path_from, destination_bucket_name, file_path_to)
    """
    storage_client = storage.Client(**kwargs)
    source_bucket = storage_client.get_bucket(source_bucket_name)
    source_blob = source_bucket.blob(source_file_path)
    destination_bucket = storage_client.get_bucket(destination_bucket_name)

    # copy to new destination
    new_blob = source_bucket.copy_blob(
        source_blob, destination_bucket, destination_blob_name
    )

    return None


def add_gcp_prefix(filepath: str, bucket_name: Optional[str] = None):
    """Adds the annoying GCP prefix!!!!!
    Args:
        filepath (str): the filepath within the bucket
        gcp_prefix (str): the bucketname
    Returns:
        filepath (str): with the gcp prefix
    Examples:
        >>> add_gcp_prefix("test", "bucket")
        gs://bucket/test
    """
    if bucket_name is not None:
        return "gs://" + str(Path(bucket_name).joinpath(filepath))
    else:
        return "gs://" + filepath


def remove_gcp_prefix(filepath: str, gcp_prefix: bool = False):
    """Adds the annoying GCP prefix!!!!!
    Args:
        filepath (str): the filepath within the bucket
        gcp_prefix (str): the bucketname
    Returns:
        filepath (str): with the gcp prefix
    Examples:
        >>> add_gcp_prefix("test", "bucket")
        gs://bucket/test
    """
    filepath = filepath.replace("gs://", "")
    if gcp_prefix:
        return str(Path(*Path(filepath).parts[1:]))
    else:
        return filepath


def get_filesystem(path: Union[str, Path]):
    path = str(path)
    if "://" in path:
        # use the fileystem from the protocol specified
        return fsspec.filesystem(path.split(":", 1)[0],requester_pays = True)
    else:
        # use local filesystem
        return fsspec.filesystem("file",requester_pays = True)


def write_geojson_to_gcp(gs_path: str, geojson_val: gpd.GeoDataFrame) -> None:
    fs = get_filesystem(gs_path)
    if geojson_val.shape[0] == 0:
        warnings.warn(f"Dataframe is empty. Saving in {gs_path}")
        with fs.open(gs_path, "w") as fh:
            json.dump(eval(geojson_val.to_json()), fh)
    else:
        with fs.open(gs_path, "wb") as fh:
            geojson_val.to_file(fh, driver="GeoJSON")


def check_requester_pays_gcp_available() -> bool:
    """
    Requester pays for GCP is available from GDAL 3.4

    rasterio issue: https://github.com/rasterio/rasterio/issues/1948
    Commit GDAL: https://github.com/OSGeo/gdal/pull/3883/commits/8d551953dea9290b21bd9747ec9ed22c81ca0409

    Returns:
        True if version >= 3.4

    """
    try:
        version_gdal = rasterio.__gdal_version__
        vnumber, vsubnumber = version_gdal.split(".")[:2]
        vnumber, vsubnumber = int(vnumber), int(vsubnumber)
        if (vnumber*100+vsubnumber) >= (3*100+4):
            return True
    except Exception as e:
        pass

    return False


REQUESTER_PAYS_AVAILABLE = check_requester_pays_gcp_available()


@contextmanager
def rasterio_open_read(tifffile:str, requester_pays:bool=True) -> rasterio.DatasetReader:
    if requester_pays and tifffile.startswith("gs"):
        if REQUESTER_PAYS_AVAILABLE:
            assert "GS_USER_PROJECT" in os.environ, \
                "'GS_USER_PROJECT' env variable not found and requester_pays=True set a project name to read rasters from the bucket" \
                "(i.e. -> export GS_USER_PROJECT='myprojectname'"

            with rasterio.open(tifffile) as src:
                yield src
        else:
            fs = fsspec.filesystem("gs", requester_pays=True)
            with fs.open(tifffile, "rb") as fh:
                with rasterio.io.MemoryFile(fh.read()) as mem:
                    yield mem.open()

    else:
        with rasterio.open(tifffile) as src:
            yield src


def read_geojson_from_gcp(gs_path: str) -> gpd.GeoDataFrame:
    fs = get_filesystem(gs_path)
    with fs.open(gs_path, "rb") as fh:
        return gpd.read_file(fh)


def write_pickle_to_gcp(gs_path: str, dict_val: dict) -> None:

    fs = get_filesystem(gs_path)
    with fs.open(gs_path, "wb") as fh:
        pickle.dump(dict_val, fh)


def read_pickle_from_gcp(gs_path:str) -> dict:
    fs = get_filesystem(gs_path)
    with fs.open(gs_path, "rb") as fh:
        my_dictionary = pickle.load(fh)

    return my_dictionary


def write_json_to_gcp(gs_path: str, dict_val: dict) -> None:
    fs = get_filesystem(gs_path)

    with fs.open(gs_path, "w") as fh:
        json.dump(dict_val, fh, cls=CustomJSONEncoder)


def read_json_from_gcp(gs_path: str) ->Dict:
    fs = get_filesystem(gs_path)
    with fs.open(gs_path, "r") as fh:
        my_dictionary = json.load(fh)

    return my_dictionary

