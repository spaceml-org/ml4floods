"""
Demo script to download some demo data files. Mainly used for testing but can also be used for other explorations.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from google.cloud import storage
from shapely.ops import cascaded_union
from src.data.config import CLASS_LAND_COPERNICUSEMSHYDRO
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

HOME = str(Path.home())


def download_data_from_bucket(
    filenames: List[str],
    destination_dir: str,
    ml_split: str = "train",
    bucket_id: Optional[str] = None,
) -> None:

    for ifile in filenames:
        bucket_id = str(Path(ifile).parts[0])

        file_name = str(Path(*Path(ifile).parts[1:]))
        save_file_from_bucket(
            bucket_id, file_name=file_name, destination_file_path=destination_dir
        )


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



def filter_land(gpddats : gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """ Filter land from pandas dataframe (land class specified in hydrology maps from CopernicusEMS) """
    isnot_land = gpddats.obj_type.apply(lambda g: g not in CLASS_LAND_COPERNICUSEMSHYDRO)

    if np.sum(isnot_land) == gpddats.shape[0]:
        return gpddats

    gpddats_notland = gpddats[isnot_land].copy()
    if gpddats_notland.shape[0] == 0:
        return gpddats_notland

    land_geometries = gpddats.geometry[~isnot_land]
    land_geometries = cascaded_union(land_geometries.tolist())

    gpddats_notland["geometry"] = gpddats_notland.geometry.apply(lambda g: g.difference(land_geometries))

    return gpddats_notland


def filter_pols(gpddats: gpd.GeoDataFrame, pol_shapely: Polygon) -> gpd.GeoDataFrame:
    """ filter pols that do not intersects pol_shapely """
    gpddats_cp = gpddats[~(gpddats.geometry.isna() | gpddats.geometry.is_empty)].copy()

    return gpddats_cp[gpddats_cp.geometry.apply(lambda g: g.intersects(pol_shapely))].reset_index().copy()


def generate_list_of_files(bucket_id: str, file_path):
    """Generate a list of files from the bucket."""
    return None


def save_file_from_bucket(bucket_id: str, file_name: str, destination_file_path: str):
    """Saves a file from a bucket

    Parameters
    ----------
    bucket_id : str
        the name of the bucket
    file_name : str
        the name of the file in bucket (include the directory)
    destination_file_path : str
        the directory of where you want to save the
        data locally (not including the filename)

    Examples
    --------

    >>> bucket_id = ...
    >>> file_name = 'path/to/file/and/file.csv'
    >>> dest = 'path/in/bucket/'
    >>> load_file_from_bucket(
        bucket_id=bucket_id,
        file_name=file_name,
        destimation_file_path=dest
    )
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
    """Saves a file from a bucket

    Parameters
    ----------
    bucket_id : str
        the name of the bucket
    file_name : str
        the name of the file in bucket (include the directory)
    destination_file_path : str
        the directory of where you want to save the
        data locally (not including the filename)

    Examples
    --------

    >>> bucket_id = ...
    >>> file_name = 'path/to/file/and/file.csv'
    >>> dest = 'path/in/bucket/'
    >>> load_file_from_bucket(
        bucket_id=bucket_id,
        file_name=file_name,
        destimation_file_path=dest
    )
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
    """Saves a file to a bucket

    Parameters
    ----------
    bucket_id : str
        the name of the bucket
    local_save_file : str
        the name of the file to save (include the directory)
    destination_file_path : str
        the directory (not including) of the bucket where
        you want to save.

    Examples
    --------

    >>> bucket_id = ...
    >>> file_name = 'path/to/file/and/file.csv'
    >>> dest = 'path/in/bucket/'
    >>> load_file_from_bucket(
        bucket_id=bucket_id,
        file_name=file_name,
        destimation_file_path=dest
    )
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
    if not Path(path).is_dir():
        raise ValueError(f"Unrecognized path: {str(Path(path))}")
    return None


def create_folder(directory: str) -> None:
    """Creates directory if doesn't exist

    params:
        directory (str): a directory to be created if
            it doesn't already exist.

        Typical usage example:

        >>> from src.data.utils import create_folder
        >>> directory = "./temp"
        >>> create_folder(directory)
    """
    try:
        Path(directory).mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"Folder '{directory}' Is Already There.")
    else:
        print(f"Folder '{directory}' is created.")


def get_files_in_directory(directory: str, suffix: str) -> List[str]:
    p = Path(directory).glob(f"*{suffix}")
    files = [str(x) for x in p if x.is_file()]
    return files


def get_files_in_bucket_directory(
    bucket_id: str, directory: str, suffix: str
) -> List[str]:
    p = Path(directory).glob(f"*{suffix}")
    files = [str(x) for x in p if x.is_file()]
    return files


def get_filenames_in_directory(directory: str, suffix: str) -> List[str]:
    p = Path(directory).glob(f"*{suffix}")
    files = [str(x.name) for x in p if x.is_file()]
    return files


def get_files_in_bucket_directory(
    bucket_id: str, directory: str, suffix: str, **kwargs
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

    files = [str(x.name) for x in blobs if str(Path(x.name).suffix) == suffix]
    return files


class CustomJSONEncoder(json.JSONEncoder):

    # TODO encode shapely.geometry objects
    # TODO encode timestamps with isoformat()
    def default(self, obj_to_encode):
        """Pandas and Numpy have some specific types that we want to ensure
        are coerced to Python types, for JSON generation purposes. This attempts
        to do so where applicable.
        """
        # Pandas dataframes have a to_json() method, so we'll check for that and
        # return it if so.
        if hasattr(obj_to_encode, 'to_json'):
            return obj_to_encode.to_json()
        # Numpy objects report themselves oddly in error logs, but this generic
        # type mostly captures what we're after.
        if isinstance(obj_to_encode, np.generic):
            return obj_to_encode.item()
        # ndarray -> list, pretty straightforward.
        if isinstance(obj_to_encode, np.ndarray):
            return obj_to_encode.tolist()
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
