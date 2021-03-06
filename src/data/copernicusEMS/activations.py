import datetime
import json
import os
import shutil
import zipfile
from glob import glob
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from requests_html import HTMLSession
from shapely.ops import cascaded_union

from src.data import utils
from src.data.config import ACCEPTED_FIELDS, RENAME_SATELLITE


def is_downloadable(url: str) -> bool:
    """ 
    Function that checks if the url contains a downloadable resource.
    Relies on requests_html package. Helper function for download_vector_cems.

    Args:
      url: A string containing the Copernicus EMS url grabbed
      for specific attributes.

    Returns:
      A boolean indicating if the url is valid
    """

    h = requests.head(url, allow_redirects=True)
    header = h.headers
    content_type = header.get('content-type').lower()
    return ('text' not in content_type or 'html' not in content_type)


def table_floods_ems(event_start_date: str = "2014-05-01") -> pd.DataFrame:
    """
    This function reads the list of EMS Rapid Mapping Activations 
    from the Copernicus Emergency Management System for Event type
    'Flood' and 'Storm' and returns a pandas.DataFrame of flood events
    organized in columns {'Code', 'Title', 'CodeDate', 'Type', 'Country'}.

    Args:
      event_start_data (str): The starting date for the span of time until 
        the present that the user would like to retrieve flood events for.
        
    Returns:
      A pandas.DataFrame of Flood events. 

    """
    ems_web_page = "https://emergency.copernicus.eu/mapping/list-of-activations-rapid"
    tables = pd.read_html(ems_web_page)[0]
    tables_floods = tables[(tables.Type == "Flood") | (tables.Type == "Storm")]
    tables_floods = tables_floods[tables_floods["Event Date"] >= event_start_date]
    tables_floods = tables_floods.reset_index()[['Act. Code', 
                                                 'Title', 
                                                 'Event Date', 
                                                 'Type', 
                                                 'Country/Terr.']]
    
    tables_floods = tables_floods.rename({"Act. Code": "Code",
                                          "Country/Terr.": "Country",
                                          "Event Date": "CodeDate"},
                                         axis=1)

    return tables_floods.set_index("Code")


def fetch_zip_files(code: str) -> List[str]:
    """
    This Function takes a unique Copernicus EMS hazard activation code and 
    retrieves the url which holds the zip files associated with that code.
    
    e.g. if cod = EMMSR502, the url holding the zip files of interest is:
    https://emergency.copernicus.eu/mapping/list-of-components/EMSR502/aemfeed
    
    Args:
      code (str): The unique Copernicus EMS code for an event of interest.
      
    Returns:
      zip_url_per_code (str): url of the zipfile location for the associated
        input code.
    """
    product_url = "https://emergency.copernicus.eu/mapping/list-of-components/" + code
    session = HTMLSession()
    r = session.get(product_url)

    zip_url_per_code = []
    for zipfile in r.html.find('a'):
        if ("zip" in zipfile.attrs['href']) and ("REFERENCE_MAP" not in zipfile.attrs['href']):
            zip_url_per_code.append("https://emergency.copernicus.eu"+zipfile.attrs['href'])
    
    return zip_url_per_code


def write_file_to_bucket(user_gcp_key_file: str, 
                         bucket_name: str, 
                         bucket_path: str, 
                         file_leaf_node: str) -> None:
    """
    Function to write file to the Google Cloud Storage Bucket.
    
    Args:
      user_gcp_key_file (str): A Google Cloud Storage personal account 
        and private key json file used to access the GCP Python API to the 
        data storage bucket.
     
    Returns:
      None. File is uploaded to the GCP bucket.
    """
    # TODO this is broken move to utils.py?
    storage_client = storage.Client.from_service_account_json(user_gcp_key_file)
    
    # Create a bucket object
    bucket = storage_client.get_bucket(bucket_name)
    
    # Construct path from bucket to file leaf node in file tree
    filename_w_path = "%s/%s" % (bucket_path, file_leaf_node)
    blob = bucket.blob(filename_w_path)
    
    # Upload file bucket object
    # TODO this is broken
    blob.upload_from_filename(local_file_name)
    
    # Notify user
    print("Upload Complete")
    
# Move to config
copernicus_ems_webscrape_data  = {"confirmation": 1,
                                  "op":  "Download file",
                                  "form_build_id": "xxxx",
                                  "form_id": "emsmapping_disclaimer_download_form"}


def download_vector_cems(zipfile_url, folder_out="CopernicusEMS"):
    """
    This function downloads the zip files from the zip file url of each
    Copernicus EMS hazard activation and saves them locally to a file
    directory folder_out under a subdirectory based off the activation code.
    
    Args:
      zipfile_url (str): Url to retrieve the downloadable zip file from 
        the Copernicus EMS Rapid Mapping Activations site for a unique hazard 
        code of the form 'EMSR[0-9]{3},' where the string EMSR is followed
        by three digits taken from 0-9.
      
      folder_out (str): The name of the directory created to hold the Copernicus 
        EMS zip file download.
    
    Returns:
      file_path_out (str): a filepath built from the folder_out handle and the
        Copernicus EMS activation code name embedded in the file.
    """  
    # Activation Code derived from zipfile_url
    act_code = os.path.basename(zipfile_url)
    
    # Create a directory called folder_out to hold zipped EMS activations
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)

    # Create a filepath from folder_out and act_code if it doesn't exist
    file_path_out = os.path.join(folder_out, act_code)
    if os.path.exists(file_path_out):
        print("\tFile %s already exists. Not downloaded" % zipfile_url)
        return file_path_out

    # Check if zipfile_url is valid before making an url request
    if is_downloadable(zipfile_url):
        r = requests.get(zipfile_url, allow_redirects=True)
    else:
        r = requests.post(zipfile_url,
                          allow_redirects=True,
                          data=copernicus_ems_webscrape_data)
       
    open(file_path_out, 'wb').write(r.content) 
    return file_path_out


def unzip_copernicus_ems(file_name : str, folder_out : str = "Copernicus_EMS_raw"):
    """
    This function unzips a single zip file from a Copernicus EMS hazard to a
    local directory folder_out with a subdirectory derived from the file handle of 
    the file prior to the extension. Copernicus EMS per code zip file may be retrieved 
    and downloaded using download_vector_cems locally to access rapid mapping products 
    as described in: 
    
    https://emergency.copernicus.eu/mapping/ems/online-manual-rapid-mapping-products
    
    Args:
      file_name (str): The zip file to be extracted
      
      folder_out (str): The local directory to store the subdirectories named by the file
        handle naming convention.
        
    Returns:
       directory_to_extract_to (str): a directory based on the file_name within folder_out.

    """
    # Create directory folder_out locally
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)

    # Create subdirectories to extract zip files to based on file name handle
    zip_ref = zipfile.ZipFile(file_name, 'r')
    directory_to_extract_to = os.path.join(folder_out, 
                                           os.path.basename(os.path.splitext(file_name)[0]))
    
    # Check to see if directories exist
    if not os.path.exists(directory_to_extract_to):
        os.mkdir(directory_to_extract_to)
    elif len(os.listdir(directory_to_extract_to)) > 0:
        return directory_to_extract_to

    # Extract
    zip_ref.extractall(directory_to_extract_to)
    zip_ref.close()

    # Remove whitespace with underscores in directory names
    for fextracted in glob(os.path.join(directory_to_extract_to,"*.*")):
        if " " not in fextracted:
            continue
        newname = fextracted.replace(" ","_")
        shutil.move(fextracted, newname)
        
    return directory_to_extract_to


# move to config later 
formats = ["%d/%m/%Y T%H:%M:%SZ",
           "%d/%m/%Y %H:%M:%SZ",
           "%d/%m/%Y T%H:%M:%S",
           "%d/%m/%Y T%H:%MZ",
           "%d/%m/%Y T%H:%M UTC",
           "%d/%m/%Y %H:%M UTC",
           "%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S",
           "%Y/%m/%d %H:%M UTC",
           "%d/%m/%Y %H:%M:%S UTC"]


def is_file_in_directory(parent_dir_of_file: str, file_extension_pattern: str) -> str:
    """
    Helper function that checks whether a file already exists in the parent
    directory.
    
    Args:
      parent_dir_of_files (str): parent directory of the shape files extracted from
        Copernicus EMS.
      string_pattern (str): keyword and file extension for the file of interest.
      
    Returns:
      A string of the file of interest if it exists in the parent directory, returns
      None otherwise.
    """
    source_file = glob(os.path.join(parent_dir_of_file, file_extension_pattern))
    if len(source_file) == 1:
        return source_file[0]
    else:
        print(f"{file_extension_pattern} not found in directory {parent_dir_of_file}")
        return 


def post_event_date_difference_is_ok(date_post_event: datetime,
                                     date_ems_code: datetime,
                                     max_date_post_event: datetime,
                                     verbose: bool = False, ) -> bool:
    """
    Function to print the datetime difference between pre-flood event and post-flood event.
    
    Args:
      date_post_event (datetime): date after a flood event.
      date_ems_code (datetime): date Copernicus EMS activation code was issued.
      max_date_post_event (datetime):
      verbose:
      
    Returns:
      None
    """
    diff_dates = date_post_event - date_ems_code
    if verbose:
        if diff_dates.days < 0:
            print("Difference between dates is negative %d" % diff_dates.days)
            return False

        elif diff_dates.days > 35:
            print("difference too big %d" % diff_dates.days)
            return False
            
        elif (max_date_post_event-date_post_event).days >= 10:
            print("difference between max date post event and min date post event too big %d" % (max_date_post_event-date_post_event).days)
            return False

    return True

def _check_hydro_ok(shapefile, expected_crs:str)->bool:
    gpd_obj = gpd.read_file(shapefile)

    if gpd_obj.crs is None:
        print(f"{shapefile} file is not georeferenced")
        return False

    area_of_interest_crs = str(gpd_obj.crs).lower()

    if area_of_interest_crs != expected_crs:
        print(f"{shapefile} and observed_event_file have different georreferencing")
        return False

    if not all(notation in ACCEPTED_FIELDS for notation in gpd_obj[COLUMN_W_CLASS_HYDRO]):
#         print(f"There are unknown fields in the {COLUMN_W_CLASS_HYDRO} column of {shapefile}: {np.unique(gpd_obj[COLUMN_W_CLASS_HYDRO])}")
        not_None_gpd_obj = [row for row in gpd_obj[COLUMN_W_CLASS_HYDRO] if row is not None]
        print(f"There are unknown fields in the {COLUMN_W_CLASS_HYDRO} column of {shapefile}: {np.unique(np.array(not_None_gpd_obj))}")
        return False

    return True


COLUMN_W_CLASS_OBSERVED_EVENT = "notation"
COLUMN_W_CLASS_HYDRO = "obj_type"

def filter_register_copernicusems(unzipped_directory: str, code_date: str, verbose: bool = False) -> Optional[Dict]:
    """
    Function that collects the following files from the unzipped directories for each Copernicus EMS
    activation code and stores them into a dictionary with additional metadata with respect to the source.
    The files of interest are the shapefiles and associated supporting files for the:
    
      1. area of interest
      2. hydrography
      3. observed event
    
    Args:
      unzipped_directory (str): The name of the directory where the extracted EMS files live
      code_date (str): string representation of the EMSR Activation code date of issuance.
      verbose (bool): boolean flag to monitor the difference in pre-event and post-event dates.
        
    Returns:
      A dictionary with the keys 'name', 'ems_code', 'timestamp', 'satellite', 'area_of_interest'
      'timestamp_ems_code', 'observed_event_file', 'area_of_interest_file.'
      None if there are inconsistencies in the metadata

    """
    # Fetch source files needed to generate floodmap - source, observed event, area of interest
    source_file = is_file_in_directory(unzipped_directory, "*_source*.dbf")
    if not source_file:
        return

    observed_event_file = is_file_in_directory(unzipped_directory, "*_observed*.shp")
    if not observed_event_file:
        return
    
    area_of_interest_file = is_file_in_directory(unzipped_directory, "*_area*.shp")
    if not area_of_interest_file:
        return

    pd_source = load_source_file(source_file)
    if pd_source is None:
        return

    product_name = os.path.basename(os.path.dirname(source_file))
    ems_code = product_name.split("_")[0]

    date_post_event = min(pd_source[pd_source.eventphase == "Post-event"]["date"])
    max_date_post_event = max(pd_source[pd_source.eventphase == "Post-event"]["date"])
    satellite_post_event = np.array(pd_source[(pd_source.eventphase == "Post-event") &
                                              (pd_source.date == date_post_event)]["source_nam"])[0]

    date_ems_code = datetime.datetime.strptime(code_date, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
     
    if not post_event_date_difference_is_ok(date_post_event, date_ems_code, max_date_post_event, verbose):
        return
    
    # Check if pre-event date precedes post-event date
    content_pre_event = {}
    if np.any(pd_source.eventphase == "Pre-event"):
        date_pre_event = max(np.array(pd_source[pd_source.eventphase == "Pre-event"]["date"]))
        satellite_pre_event = np.array(pd_source[(pd_source.eventphase == "Pre-event") &
                                                 (pd_source.date == date_pre_event)]["source_nam"])[0]

        content_pre_event["satellite_pre_event"] = satellite_pre_event
        content_pre_event["timestamp_pre_event"] = date_pre_event
        if (date_post_event - date_pre_event).days < 0 and verbose:
            print("Date pre event %s is after date post event %s" % (date_pre_event.strftime("%Y-%m-%d"),
                                                                     date_post_event.strftime("%Y-%m-%d")))
            return

    # Filter content of shapefile
    pd_geo = gpd.read_file(observed_event_file)
    if np.any(pd_geo["event_type"] != '5-Flood') and verbose:
        print(f"{observed_event_file} Event type is not Flood {np.unique(pd_geo['event_type'])}")
        return

    if not all(notation in ACCEPTED_FIELDS for notation in pd_geo[COLUMN_W_CLASS_OBSERVED_EVENT]):
        print(f"There are unknown fields in the {COLUMN_W_CLASS_OBSERVED_EVENT} column of {observed_event_file}: {np.unique(pd_geo[COLUMN_W_CLASS_OBSERVED_EVENT])}")
        return

    if pd_geo.crs is None:
        print(f"{observed_event_file} file is not georeferenced")
        return

    str_crs = str(pd_geo.crs).lower()

    if not isinstance(satellite_post_event, str) and verbose:
        print("Satellite post event: {} is not a string!".format(satellite_post_event))
        return

    if satellite_post_event in RENAME_SATELLITE:
        satellite_post_event = RENAME_SATELLITE[satellite_post_event]

    area_of_interest = gpd.read_file(area_of_interest_file)

    if area_of_interest.crs is None:
        print(f"{area_of_interest_file} file is not georeferenced")
        return

    area_of_interest_crs = str(area_of_interest.crs).lower()

    if area_of_interest_crs != str_crs:
        print(f"{area_of_interest_file} and {observed_event_file} have different georreferencing")
        return

    if area_of_interest_crs != "epsg:4326":
        area_of_interest.to_crs(crs="epsg:4326", inplace=True)

    # Save pol of area of interest in epsg:4326 (lat/lng)
    area_of_interest_pol = cascaded_union(area_of_interest["geometry"])
    
    if "obj_desc" in pd_geo:
        event_type = np.unique(pd_geo.obj_desc)
        if len(event_type) > 1:
            print("Multiple event types within shapefile {}".format(event_type))
        event_type = event_type[0]
    else:
        event_type = "NaN"

    crs_code_space, crs_code = str_crs.split(":")
    
    register = {
        'event id': product_name,
        'layer name': os.path.basename(os.path.splitext(observed_event_file)[0]),
        'event type': event_type,
        'satellite date': date_post_event,
        'country': "NaN",
        'satellite': satellite_post_event,
        'bounding box': get_bbox(pd_geo),
        'reference system': {
            'code space': crs_code_space,
            'code': crs_code
        },
        'abstract': "NaN",
        'purpose': "NaN",
        'source': 'CopernicusEMS',
        "area_of_interest_polygon" : area_of_interest_pol,
        # CopernicusEMS specific fields
        "observed_event_file": os.path.basename(observed_event_file),
        "area_of_interest_file": os.path.basename(area_of_interest_file),
        "ems_code": ems_code,
    }
    
    register.update(content_pre_event)

    name_possibilities = ["_hydrographyA_", "_hydrography_a"]
    for name_pos in name_possibilities:
        hydrology_file_as = glob(os.path.join(unzipped_directory, f"*{name_pos}*.shp"))
        if len(hydrology_file_as) == 1:
            if not _check_hydro_ok(hydrology_file_as[0], expected_crs=str_crs):
                return
            register["hydrology_file"] = os.path.basename(hydrology_file_as[0])

    name_possibilities = ["_hydrographyL_", "_hydrography_l"]
    for name_pos in name_possibilities:
        hydrology_file_l = glob(os.path.join(unzipped_directory, f"*{name_pos}*.shp"))
        if len(hydrology_file_l) == 1:
            if not _check_hydro_ok(hydrology_file_l[0], expected_crs=str_crs):
                return
            register["hydrology_file_l"] = os.path.basename(hydrology_file_l[0])

    return register


def filter_register_copernicusems_gcp(unzipped_directory: str, code_date: str, verbose: bool = False) -> Optional[Dict]:
    """
    Function that collects the following files from the unzipped directories for each Copernicus EMS
    activation code and stores them into a dictionary with additional metadata with respect to the source.
    The files of interest are the shapefiles and associated supporting files for the:
    
      1. area of interest
      2. hydrography
      3. observed event
    
    Args:
      unzipped_directory (str): The name of the directory where the extracted EMS files live
      code_date (str): string representation of the EMSR Activation code date of issuance.
      verbose (bool): boolean flag to monitor the difference in pre-event and post-event dates.
        
    Returns:
      A dictionary with the keys 'name', 'ems_code', 'timestamp', 'satellite', 'area_of_interest'
      'timestamp_ems_code', 'observed_event_file', 'area_of_interest_file.'
      None if there are inconsistencies in the metadata

    """
#   sample path: gs://ml4cc_data_lake/0_DEV/0_Raw/WorldFloods/copernicus_ems/copernicus_ems_unzip/EMSR340/09UNTERGREUTH

    aoi_files = utils.get_files_in_bucket_directory_gs(gs_path=unzipped_directory)
    
    source_files = []
    observed_event_files = []
    area_of_interest_files = []
    hydrology_files_a = []
    hydrology_files_l = []
    
    for f in aoi_files:
        if f.endswith(".dbf") and "_source" in os.path.basename(f):
            source_files.append(f)
        elif f.endswith(".shp") and "_observed" in os.path.basename(f):
            observed_event_files.append(f)
        elif f.endswith(".shp") and "_area" in os.path.basename(f):
            area_of_interest_files.append(f)
        elif f.endswith(".shp") and ("_hydrographyA_" in os.path.basename(f) or "_hydrography_a" in os.path.basename(f)):
            hydrology_files_a.append(f)
        elif f.endswith(".shp") and ("_hydrographyL_" in os.path.basename(f) or "_hydrography_l" in os.path.basename(f)):
            hydrology_files_l.append(f)

    register_list = []
    
    for i_source_file in source_files:
        # continue to the next source file if the current observed_event_file file doesn't exist
        observed_event_file = [f for f in observed_event_files if i_source_file.split("_source")[0] in f]
        observed_event_file = None if len(observed_event_file) < 1 else observed_event_file[0]
        if observed_event_file is None:
            continue

        # continue to the next source file if the current area_of_interest_file file doesn't exist
        area_of_interest_file = [f for f in area_of_interest_files if i_source_file.split("_source")[0] in f]
        area_of_interest_file = None if len(area_of_interest_file) < 1 else area_of_interest_file[0]
        if area_of_interest_file is None:
            continue

        # continue to the next source file if the current source file is not readable
        pd_source = load_source_file(i_source_file)
        if pd_source is None:
            continue

        product_name = os.path.basename(i_source_file.replace("_source", "")).split(".")[0]
        ems_code = os.path.basename(unzipped_directory)

        date_post_event = min(pd_source[pd_source.eventphase == "Post-event"]["date"])
        max_date_post_event = max(pd_source[pd_source.eventphase == "Post-event"]["date"])
        satellite_post_event = np.array(pd_source[(pd_source.eventphase == "Post-event") &
                                                  (pd_source.date == date_post_event)]["source_nam"])[0]

        date_ems_code = datetime.datetime.strptime(code_date, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)

        if not post_event_date_difference_is_ok(date_post_event, date_ems_code, max_date_post_event, verbose):
            continue

        # Check if pre-event date precedes post-event date
        content_pre_event = {}
        if np.any(pd_source.eventphase == "Pre-event"):
            date_pre_event = max(np.array(pd_source[pd_source.eventphase == "Pre-event"]["date"]))
            satellite_pre_event = np.array(pd_source[(pd_source.eventphase == "Pre-event") &
                                                     (pd_source.date == date_pre_event)]["source_nam"])[0]

            content_pre_event["satellite_pre_event"] = satellite_pre_event
            content_pre_event["timestamp_pre_event"] = date_pre_event
            if (date_post_event - date_pre_event).days < 0 and verbose:
                print("Date pre event %s is after date post event %s" % (date_pre_event.strftime("%Y-%m-%d"),
                                                                         date_post_event.strftime("%Y-%m-%d")))
                continue

        # Filter content of shapefile
        pd_geo = gpd.read_file(observed_event_file)
        if np.any(pd_geo["event_type"] != '5-Flood') and verbose:
            print(f"{observed_event_file} Event type is not Flood {np.unique(pd_geo['event_type'])}")
            continue

        if not all(notation in ACCEPTED_FIELDS for notation in pd_geo[COLUMN_W_CLASS_OBSERVED_EVENT]):
            print(f"There are unknown fields in the {COLUMN_W_CLASS_OBSERVED_EVENT} column of {observed_event_file}: {np.unique(pd_geo[COLUMN_W_CLASS_OBSERVED_EVENT])}")
            continue

        if pd_geo.crs is None:
            print(f"{observed_event_file} file is not georeferenced")
            continue

        str_crs = str(pd_geo.crs).lower()

        if not isinstance(satellite_post_event, str) and verbose:
            print("Satellite post event: {} is not a string!".format(satellite_post_event))
            continue

        if satellite_post_event in RENAME_SATELLITE:
            satellite_post_event = RENAME_SATELLITE[satellite_post_event]

        area_of_interest = gpd.read_file(area_of_interest_file)

        if area_of_interest.crs is None:
            print(f"{area_of_interest_file} file is not georeferenced")
            continue

        area_of_interest_crs = str(area_of_interest.crs).lower()

        if area_of_interest_crs != str_crs:
            print(f"{area_of_interest_file} and {observed_event_file} have different georreferencing")
            continue

        # TODO: Check if this "epsg:4326" is a constant value or should be a config variable
        if area_of_interest_crs != "epsg:4326":
            area_of_interest.to_crs(crs="epsg:4326", inplace=True)

        # Save pol of area of interest in epsg:4326 (lat/lng)
        area_of_interest_pol = cascaded_union(area_of_interest["geometry"])

        if "obj_desc" in pd_geo:
            event_type = np.unique(pd_geo.obj_desc)
            if len(event_type) > 1:
                print("Multiple event types within shapefile {}".format(event_type))
            event_type = event_type[0]
        else:
            event_type = "NaN"

        crs_code_space, crs_code = str_crs.split(":")

        register = {
            'event_id': product_name,
            'layer_name': os.path.basename(os.path.splitext(observed_event_file)[0]),
            'event_type': event_type,
            'satellite_date': date_post_event,
            'country': "NaN",
            'satellite': satellite_post_event,
            'bounding_box': get_bbox(pd_geo),
            'reference_system': {
                'code_space': crs_code_space,
                'code': crs_code
            },
            'abstract': "NaN",
            'purpose': "NaN",
            'source': 'CopernicusEMS',
            "area_of_interest_polygon" : area_of_interest_pol,
            # CopernicusEMS specific fields
            "observed_event_file": os.path.basename(observed_event_file),
            "area_of_interest_file": os.path.basename(area_of_interest_file),
            "ems_code": ems_code,
            "aoi_code": product_name.split("_")[1],
        }

        register.update(content_pre_event)
        
        hydrology_file_a = [f for f in hydrology_files_a if i_source_file.split("_source")[0] in f]
        hydrology_file_a = None if len(hydrology_file_a) < 1 else hydrology_file_a[0]
        if hydrology_file_a is not None:
            if not _check_hydro_ok(hydrology_file_a, expected_crs=str_crs):
                continue
            register["hydrology_file_a"] = os.path.basename(hydrology_file_a)

        hydrology_file_l = [f for f in hydrology_files_l if i_source_file.split("_source")[0] in f]
        hydrology_file_l = None if len(hydrology_file_l) < 1 else hydrology_file_l[0]
        if hydrology_file_l is not None:
            if not _check_hydro_ok(hydrology_file_l, expected_crs=str_crs):
                continue
            register["hydrology_file_l"] = os.path.basename(hydrology_file_l)

        register_list.append(register)
            
    return register_list


def parse_date_messy(date_list):
    """
    Helper function for load_source_file
    """
    for fm in formats:
        try:
            #date_list[0] = date_list[0].replace("119/09", "19/09").replace("05/24", "24/05")
            time_part = date_list[1].replace(".", ":").replace(";", ":").replace("UCT", "UTC")
            if time_part == "Not Applicable":
                time_part = "00:00"
            date_post_event = datetime.datetime.strptime(date_list[0] + ' ' + time_part, fm)
            if date_post_event.tzname() is None:
                date_post_event = date_post_event.replace(tzinfo=datetime.timezone.utc)
            return date_post_event
        except ValueError:
            pass

    return None


def load_source_file(source_file, filter_event_dates=True, verbose=False):
    """
    Helper function for filter_register_copernicusems
    """

    pd_source = gpd.read_file(source_file)

    # pd_source = Dbf5(source_file).to_dataframe()

    if "eventphase" not in pd_source.columns or not np.any(pd_source.eventphase == "Post-event"):
        if verbose:
            print("\t eventphase not found or not Post-event in source file")
        return

    dates_formated = []
    for d in pd_source.itertuples():
        date = parse_date_messy([d.src_date, d.source_tm])
        dates_formated.append(date)
    pd_source["date"] = dates_formated
    if filter_event_dates:
        pd_source = pd_source[(pd_source.eventphase == "Post-event") | (pd_source.eventphase == "Pre-event")]

        if np.any(pd.isna(pd_source.date)):
            if verbose:
                print("\t There are event phase files with undefined dates")

            pd_source = pd_source[~pd.isna(pd_source.date)]

        if not np.any(pd_source.eventphase == "Post-event"):
            if verbose:
                print("\t Post-event removed after filtering")
            return

    return pd_source


def generate_floodmap_gcp(metadata_floodmap: Dict, folder_files:str, filterland:bool=True) -> gpd.GeoDataFrame:
    """ Generates a floodmap with the joined info of the hydro and flood. 
    Works the same way as the generate_floodmap() function but reads files from the bucket."""
    
    area_of_interest_file = os.path.join(folder_files, 
                                         metadata_floodmap["ems_code"], 
                                         metadata_floodmap["aoi_code"], 
                                         metadata_floodmap["area_of_interest_file"])
    area_of_interest = gpd.read_file(area_of_interest_file)
    
    area_of_interest_pol = cascaded_union(area_of_interest["geometry"])

    observed_event_file = os.path.join(folder_files, 
                                       metadata_floodmap["ems_code"], 
                                       metadata_floodmap["aoi_code"], 
                                       metadata_floodmap["observed_event_file"])
    mapdf = utils.filter_pols(gpd.read_file(observed_event_file), area_of_interest_pol)
    assert mapdf.shape[0] > 0, f"No polygons within bounds for {metadata_floodmap}"
    
    floodmap = gpd.GeoDataFrame({"geometry": mapdf.geometry},
                                crs=mapdf.crs)
    floodmap["w_class"] = mapdf[COLUMN_W_CLASS_OBSERVED_EVENT]
    floodmap["source"] = "flood"
    
    if "hydrology_file_a" in metadata_floodmap:
        hydrology_file_a = os.path.join(folder_files, 
                                        metadata_floodmap["ems_code"], 
                                        metadata_floodmap["aoi_code"],  
                                        metadata_floodmap["hydrology_file_a"])
        
        mapdf_hydro = utils.filter_pols(gpd.read_file(hydrology_file_a), area_of_interest_pol)

        mapdf_hydro = utils.filter_land(mapdf_hydro) if filterland and (mapdf_hydro.shape[0] > 0) else mapdf_hydro
        
        if mapdf_hydro.shape[0] > 0:
            mapdf_hydro["source"] = "hydro"
            mapdf_hydro = mapdf_hydro.rename({COLUMN_W_CLASS_HYDRO: "w_class"}, axis=1)
            mapdf_hydro = mapdf_hydro[["geometry", "w_class", "source"]].copy()
            floodmap = pd.concat([floodmap, mapdf_hydro], axis=0, ignore_index=True)
        
    # Add "hydrology_file_l"?? must be handled in later in create_gt.compute_water function
    if "hydrology_file_l" in metadata_floodmap:
        hydrology_file_l = os.path.join(folder_files, 
                                        metadata_floodmap["ems_code"], 
                                        metadata_floodmap["aoi_code"],  
                                        metadata_floodmap["hydrology_file_l"])
        
        mapdf_hydro = utils.filter_pols(gpd.read_file(hydrology_file_l), area_of_interest_pol)
        if mapdf_hydro.shape[0] > 0:
            mapdf_hydro["source"] = "hydro_l"
            mapdf_hydro = mapdf_hydro.rename({COLUMN_W_CLASS_HYDRO: "w_class"}, axis=1)
            mapdf_hydro = mapdf_hydro[["geometry", "w_class", "source"]].copy()
            floodmap = pd.concat([floodmap, mapdf_hydro], axis=0, ignore_index=True)
    
    # Concat area of interest
    area_of_interest["source"] = "area_of_interest"
    area_of_interest["w_class"] = "area_of_interest"
    area_of_interest = area_of_interest[["geometry", "w_class", "source"]].copy()
    floodmap = pd.concat([floodmap, area_of_interest], axis=0, ignore_index=True)

    assert floodmap.crs is not None, "Unexpected error. floodmap is not georreferenced!"
    
    floodmap.loc[floodmap.w_class.isna(), 'w_class'] = "Not Applicable"

    return floodmap


def generate_floodmap(metadata_floodmap: Dict, folder_files:str, filterland:bool=True) -> gpd.GeoDataFrame:
    """ Generates a floodmap with the joined info of the hydro and flood. """

    area_of_interest = gpd.read_file(os.path.join(folder_files, metadata_floodmap["event id"] ,
                                                  metadata_floodmap["area_of_interest_file"]))

    area_of_interest_pol = cascaded_union(area_of_interest["geometry"])

    mapdf = utils.filter_pols(gpd.read_file(os.path.join(folder_files, metadata_floodmap["event id"] ,
                                                         metadata_floodmap["observed_event_file"])),
                              area_of_interest_pol)
    assert mapdf.shape[0] > 0, f"No polygons within bounds for {metadata_floodmap}"

    floodmap = gpd.GeoDataFrame({"geometry": mapdf.geometry},
                                crs=mapdf.crs)
    floodmap["w_class"] = mapdf[COLUMN_W_CLASS_OBSERVED_EVENT]
    floodmap["source"] = "flood"

    if "hydrology_file" in metadata_floodmap:
        mapdf_hydro = utils.filter_pols(gpd.read_file(os.path.join(folder_files, metadata_floodmap["event id"] ,
                                                                   metadata_floodmap["hydrology_file"])),
                                        area_of_interest_pol)
        mapdf_hydro = utils.filter_land(mapdf_hydro) if filterland and (mapdf_hydro.shape[0] > 0) else mapdf_hydro
        if mapdf_hydro.shape[0] > 0:
            mapdf_hydro["source"] = "hydro"
            mapdf_hydro = mapdf_hydro.rename({COLUMN_W_CLASS_HYDRO: "w_class"}, axis=1)
            mapdf_hydro = mapdf_hydro[["geometry", "w_class", "source"]].copy()
            floodmap = pd.concat([floodmap, mapdf_hydro], axis=0, ignore_index=True)

    # Add "hydrology_file_l"?? must be handled in later in create_gt.compute_water function
    if "hydrology_file_l" in metadata_floodmap:
        mapdf_hydro = utils.filter_pols(gpd.read_file(os.path.join(folder_files, metadata_floodmap["event id"] ,
                                                                   metadata_floodmap["hydrology_file_l"])),
                                        area_of_interest_pol)
        if mapdf_hydro.shape[0] > 0:
            mapdf_hydro["source"] = "hydro_l"
            mapdf_hydro = mapdf_hydro.rename({COLUMN_W_CLASS_HYDRO: "w_class"}, axis=1)
            mapdf_hydro = mapdf_hydro[["geometry", "w_class", "source"]].copy()
            floodmap = pd.concat([floodmap, mapdf_hydro], axis=0, ignore_index=True)

    # Concat area of interest
    area_of_interest["source"] = "area_of_interest"
    area_of_interest["w_class"] = "area_of_interest"
    area_of_interest = area_of_interest[["geometry", "w_class", "source"]].copy()
    floodmap = pd.concat([floodmap, area_of_interest], axis=0, ignore_index=True)

    assert floodmap.crs is not None, "Unexpected error. floodmap is not georreferenced!"

    # TODO set CRS of floodmap? check it is still OK

    floodmap.loc[floodmap.w_class.isna(), 'w_class'] = "Not Applicable"

    return floodmap


def generate_polygon(bbox):
    """
    Generates a list of coordinates: [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x1,y1]]
    """
    return [[bbox[0],bbox[1]],
             [bbox[2],bbox[1]],
             [bbox[2],bbox[3]],
             [bbox[0],bbox[3]],
             [bbox[0],bbox[1]]]

def get_bbox(pd_geo: gpd.GeoDataFrame) -> Dict:
    """
    This function is a helper function of processing_register
    This function takes a geopandas dataframe 
    
    """
    bounds = pd_geo.bounds
    minx = np.min(bounds.minx)
    miny = np.min(bounds.miny)
    maxx = np.max(bounds.maxx)
    maxy = np.max(bounds.maxy)

    return {"west": minx, "east": maxx, "north": maxy, "south": miny}


COPY_FORMATS = ["dbf", "prj", "shp", "shx"]






