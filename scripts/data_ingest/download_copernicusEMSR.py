from pyprojroot import here
import sys
sys.path.append(str(here()))

from google.cloud import storage
from ml4floods.data.copernicusEMS import activations

import requests
from io import BytesIO
from typing import Dict
from zipfile import ZipFile, is_zipfile
import tqdm
import os
import fsspec


copernicus_ems_webscrape_data  = {"confirmation": 1,
                                  "op":  "Download file",
                                  "form_build_id": "xxxx",
                                  "form_id":"emsmapping_disclaimer_download_form"}




def load_ems_zipfiles_to_gcp(url_of_zip: str,
                             bucket_id: str,
                             path_to_write_to: str,
                             copernicus_ems_web_structure: Dict):
    """
    Function to retrieve zipfiles from Copernicus EMS based on 
    EMSR code and webpage for each individual code and save it
    to a Google Cloud Storage bucket.
    
    Example:
    https://emergency.copernicus.eu/mapping/list-of-components/EMSR502
    
    is the page for EMSR code = 502, and contains the list of zipfiles
    for this particular code. May include multiple areas of interests
    for the same code.
    
    Requirements:
    requests
    google.cloud.storage
    io.BytesIO
    
    Args:
      url_of_zip (str): url of the zip file for a given ESMR code.
      bucket_id (str): name of Google Cloud Storage bucket
      path_to_write_to (str): path to write to in Google Cloud Storage bucket
      copernicus_ems_web_structure (Dict): dictionary derived from json of 
          Copernicus EMSR url for a single code. Potentially brittle.
      
    Returns:
      None
    """
    client = storage.Client()
    bucket = client.get_bucket(bucket_id)
    blob = bucket.blob(path_to_write_to)
    
    r = requests.post(url_of_zip, allow_redirects=True, data=copernicus_ems_web_structure)
    f = BytesIO(r.content)
    blob.upload_from_string(f.read(), content_type="application/zip")
    

def extract_ems_zip_files_gcp(bucket_id: str, file_path_to_zip: str, file_path_to_unzip: str):
    """
    Function to extract Copernicus EMS zip files from individual EMSR code
    page.
    
    Args:
      bucket_id (str): name of Google Cloud Storage bucket.
      file_path_to_zip (str): name of file path in bucket leading to zip file.
      file_path_to_unzip (str): name of file path in bucket leading to extracted files.
    """
    client = storage.Client()
    bucket = client.get_bucket(bucket_id)
    blob_to_zip = bucket.blob(file_path_to_zip)
    
    zip_file_from_gcpbucket = blob_to_zip.download_as_bytes()
    f_from_gcp = BytesIO(zip_file_from_gcpbucket)
    
    zipdict = {}
    input_zip = ZipFile(f_from_gcp)
    if is_zipfile(f_from_gcp):
        for name in input_zip.namelist():
            name_dest = file_path_to_unzip + "/" + name
            if fs.exists(f"gs://{bucket_id}/{name_dest}"):
                continue
            if 'area' in name or 'hydrography' in name or 'observed' in name or 'source' in name:
                zipdict[name] = input_zip.read(name)
                blob_to_unzipped = bucket.blob(name_dest)
                blob_to_unzipped.upload_from_string(zipdict[name])


def main():
    bucket_id = 'ml4cc_data_lake'
    path_to_write_zip = '0_DEV/0_Raw/WorldFloods/copernicus_ems/copernicus_ems_zip'
    path_to_write_unzip = '0_DEV/0_Raw/WorldFloods/copernicus_ems/copernicus_ems_unzip'
    import argparse

    parser = argparse.ArgumentParser('Download Copernicus EMS')
    parser.add_argument('--event_start_date', default="2015-07-01",
                        help="Event start date to consider when fetching the images from CEMS")
    args = parser.parse_args()
    
    # fetch Copernicus EMSR codes from Copernicus EMS activations page
    # pandas DataFrame of activations table
    table_activations_ems = activations.table_floods_ems(event_start_date=args.event_start_date)
    
    # convert code index to a list
    emsr_codes = table_activations_ems.index.to_list()

    # retrieve zipfile urls for each activation EMSR code
    print(f"Generating Copernicus EMSR codes to fetch:")
    with tqdm.tqdm(emsr_codes) as pbar:
        for emsr in pbar:
            pbar.set_description(f"Code: {emsr}")
            zip_files_activation_url_list = activations.fetch_zip_file_urls(emsr)
            for zip_url in zip_files_activation_url_list:
                aoi = zip_url.split('_')[1]
                path_to_write_unzip_bucket = f"{path_to_write_unzip}/{emsr}/{aoi}"
                name_zip = os.path.basename(zip_url)
                path_to_write_zip_bucket = f"{path_to_write_zip}/{emsr}/{aoi}/{name_zip}"
                if not fs.exists(f"gs://{bucket_id}/{path_to_write_zip_bucket}"):
                    load_ems_zipfiles_to_gcp(zip_url,
                                             bucket_id,
                                             path_to_write_zip_bucket,
                                             copernicus_ems_webscrape_data)

                extract_ems_zip_files_gcp(bucket_id,
                                         path_to_write_zip_bucket,
                                         path_to_write_unzip_bucket)
    
    
if __name__ == "__main__":
    fs = fsspec.filesystem("gs")
    main()
