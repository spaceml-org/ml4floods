import glob
import os
import pandas as pd
from google.cloud import storage
from os import path
from typing import NamedTuple, List
from collections import namedtuple
import pandas as pd

# Helpful trick for loading the directories correction
import sys, os
from pyprojroot import here
from pathlib import Path


def construct_worldfloods_public_filepaths() -> NamedTuple:
    """
    This function takes the pre-existing worldfloods public directory
    and constructs the paths to train, test, and validate for a specific
    data (Sentinel-2, Ground Truth Labels, Sentinel-2 metadata) and
    returns the file paths as a namedtuple based on the file type and
    whether the data is located in train, test, or val.

    Args:
      None
    Returns:
      NamedTuples based on 'train', 'test', 'val' split of file paths associated
      with Sentinel-2, Sentinel-2 metadata, and ground truth.

    """
    # Constructing the filepaths
    prefix = "gs://"
    path_bucket = "worldfloods/public"

    sub_dir_train = "train"
    sub_dir_test = "test"
    sub_dir_val = "val"

    path_train = os.path.join(path_bucket, sub_dir_train) #worldfloods/public/train/S2
    path_test = os.path.join(path_bucket, sub_dir_test)
    path_val = os.path.join(path_bucket, sub_dir_val)


    Bucket_file_paths = namedtuple('Bucket_file_path', ['split_group', 'S2', 'meta', 'gt'])

    paths_train = Bucket_file_paths(split_group = 'train',
                                 S2 = os.path.join(path_train, "S2"),\
                                 meta = os.path.join(path_train,"meta"),\
                                 gt = os.path.join(path_train, "gt"))

    paths_test = Bucket_file_paths(split_group = 'test',
                                   S2 = os.path.join(path_test, "S2"),\
                                   meta = os.path.join(path_test, "meta"),\
                                   gt = os.path.join(path_test, "gt"))

    paths_val = Bucket_file_paths(split_group = 'gt',
                                 S2 = os.path.join(path_val, 'S2'),\
                                 meta = os.path.join(path_val, 'meta'),\
                                 gt = os.path.join(path_val, 'gt'))

    return paths_train, paths_test, paths_val


def get_file_identifier_from_s2(paths: NamedTuple) -> List[str]:
    """
    This function takes in a NamedTuple of file paths to retrieve
    google.cloud.storage.blob.Blob objects associated with Sentinel-2 files
    in the WorldFloods bucket and returns a list of string file_identifiers.

    Args:
      paths (NamedTuple):
        Formatted by Bucket_file_paths = namedtuple('Bucket_file_paths', ['split', 'S2', 'meta', 'gt'])
        where 'split' can be {'train', 'test', 'val'}, and 'S2', 'meta', and 'gt' are file paths.

    Returns
      A list of strings representing the filename identifier for the data
      without the file extension.

    """
    # Instantiates a client
    storage_client = storage.Client()

    bucket_name = 'ml4floods'

    # Get GCS bucket
    bucket = storage_client.get_bucket(bucket_name)

    blobs_s2 = list(bucket.list_blobs(prefix=paths.S2))

    file_identifier = []
    for blob in blobs_s2:
        try:
            num = blob.name.count('/')
            string = blob.name.split('/')[num]
            string = string.split('.')
            if string != "":
                file_identifier.append(string[0])

        except:
            print("An exception occurred")

    return file_identifier

def uri_table_from_file_identifier(file_identifier: List[str], file_paths: NamedTuple) -> pd.DataFrame:
    """
    This Function takes as input the list of file identifiers retrieved for Sentinel-2
    data from the worldfloods bucket on Google Cloud. The file identifiers are then
    used to search for the corresponding file name in the ground truth and Sentinel-2
    metadata subdirectories in worldsfloods/public for either 'train', 'test', or 'val'.

    Args:
      file_identifier (List[str]): consisting of the file names of data without
      the file extensions.

      file_path (NamedTuple):

      split_group (str): A string indicating if it is 'train', 'test', or split.
    Returns:
      A pandas DataFrame object containing the file identifier, Sentinel-2
      (S2) image URI, S2 metadata, and "ground truth" image output, or
      data labeling automation output, from S2cloudless.

    """
    # Instantiates a client
    storage_client = storage.Client()

    bucket_name = 'ml4floods'

    # Get GCS bucket
    bucket = storage_client.get_bucket(bucket_name)
    list_gt_paths = []
    list_S2_meta_paths = []
    list_S2_paths = []
    for item in file_identifier:
        file_gt_full_path = str(file_paths.gt) + '/' + item + ".tif"
        file_S2_meta_full_path = str(file_paths.meta) + '/' + item + ".json"
        if bucket.blob(file_gt_full_path).exists() & bucket.blob(file_S2_meta_full_path).exists():
            list_gt_paths.append(prefix + file_gt_full_path)
            list_S2_meta_paths.append(prefix + file_S2_meta_full_path)
            list_S2_paths.append(prefix + str(file_paths.S2) + '/' + item + ".tiff")

    df_uri = pd.DataFrame(list(zip(file_identifier, list_S2_paths, list_S2_meta_paths, list_gt_paths)),
                      columns = ['file_identifier', 'S2_uri', 'S2_meta_uri', 'gt_uri'])
    df_uri = df_uri.assign(Split = file_paths.split_group)

    return df_uri

def main():
    """
    Main function to call helper functions to generate paths for each type of data in the bucket,
    derive file_identifier from Sentinel-2, search for files in the S2 metadata and ground truth
    data for corresponding filename and organize the URIs for the data using Pandas.
    """


    # spyder up to find the root
    root = here(project_files=[".here"])

    # append to path
    sys.path.append(str(here()))

    # Construct file paths for worldfloods public Google Cloud storage bucket folder
    paths_train, paths_test, paths_val = construct_worldfloods_public_filepaths()


    # Extract the file identifiers/names of files without the file extension to
    # utilize for search.
    file_id_train = get_file_identifier_from_s2(paths_train)
    file_id_test = get_file_identifier_from_s2(paths_test)
    file_id_val = get_file_identifier_from_s2(paths_val)

    # Create a pandas DataFrame containing image URIs
    df_train = uri_table_from_file_identifier(file_id_train, paths_train)
    df_test = uri_table_from_file_identifier(file_id_test, paths_test)
    df_val = uri_table_from_file_identifier(file_id_val, paths_val)

    # Concatenate the pandas DataFrame along axis = 0 to
    # add testing and validation data in the same dataFrame.
    df_train_test_val = pd.concat([df_train, df_test, df_val], axis=0)

    # Save the dataframe as a comma separated value (.csv) file
    # for furtue
    df_train_test_val.to_csv(Path(root).joinpath("datasets/trials/image_meta_table.csv"))  #save it to a bucket


if __name__ == "___main__":
  main()
