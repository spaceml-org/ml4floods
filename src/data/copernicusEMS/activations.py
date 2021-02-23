from requests_html import HTMLSession
import pandas as pd
import requests
import os
import time
import shutil
from typing import List
import zipfile
from glob import glob
from simpledbf import Dbf5
import numpy as np
import datetime



def is_downloadable(url: str) -> bool:
    """ 
    Function that checks if the url contains a downloadable resource

    Args:
      url:
        A string containing the Copernicus EMS url grabbed
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
    'Flood' and 'Storm' and returns a pandas.DataFrame of flood events.

    Args:
      event_start_data (str): 
    Returns:
     A pandas.DataFrame of Flood events. 

    """
    # FIXME: Why are the inputs hard-coded?!!!!
    # lines 28 - 30 exist in copernicusEMS_web_crawler.py
    ems_web_page = "https://emergency.copernicus.eu/mapping/list-of-activations-rapid"
    tables = pd.read_html(ems_web_page)[0]
    tables_floods = tables[(tables.Type == "Flood") | (tables.Type == "Storm")]
    tables_floods = tables_floods[tables_floods["Event Date"] >= event_start_date]
    tables_floods = tables_floods.reset_index()[['Act. Code', 'Title', 'Event Date', 'Type', 'Country/Terr.']]
    tables_floods = tables_floods.rename({"Act. Code": "Code", "Country/Terr.": "Country", "Event Date": "CodeDate"},
                                         axis=1)

    return tables_floods

def download_vector_cems(url_zip_file, folder_out="CopernicusEMS"):
    """
    Args:
      url_zip_file (str): 
    """
    name_zip = os.path.basename(url_zip_file)

    if not os.path.exists(folder_out):
        os.mkdir(folder_out) # creates a directory called CopernicusEMS to hold data

    file_path_out = os.path.join(folder_out, name_zip)
    if os.path.exists(file_path_out):
        logging.info("\tFile %s exists will not be download"%url_zip_file)
        return file_path_out

    if is_downloadable(url_zip_file):
        r = requests.get(url_zip_file, allow_redirects=True)
    else: 
        data  = {"confirmation": 1,
                 "op":  "Download file",
                 "form_build_id": "xxxx",
                 "form_id": "emsmapping_disclaimer_download_form"}
        
        r = requests.post(url_zip_file,
                         allow_redirects=True, data=data)
       
    open(file_path_out, 'wb').write(r.content) 
    return file_path_out

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
    pd_source = Dbf5(source_file).to_dataframe()

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

def unzip_copernicus_ems(file_name, folder_out= "Copernicus_EMS_raw"):
    """
    This function requires the data to be stored in the directory data_root
    with the file directory structure set by the worldfloods google cloud
    bucket. CopernicusEMS is a zip file that containes 

    """
    if not os.path.exists(folder_out):
        os.mkdir(folder_out) # creates a directory called CopernicusEMS to hold data

    zip_ref = zipfile.ZipFile(file_name, 'r')
    directory_to_extract_to = os.path.join(folder_out, os.path.basename(os.path.splitext(file_name)[0]))
    if not os.path.exists(directory_to_extract_to):
        os.mkdir(directory_to_extract_to)

    zip_ref.extractall(directory_to_extract_to)
    zip_ref.close()

    # remove underscores
    for fextracted in glob(os.path.join(directory_to_extract_to,"*.*")):
        if " " not in fextracted:
            continue
        newname = fextracted.replace(" ","_")
        shutil.move(fextracted, newname)
        
    return directory_to_extract_to


def fetch_zip_files(code: str) -> List[str]:
    """
    FILL MEEE
    """
    product_url = "https://emergency.copernicus.eu/mapping/list-of-components/" + code
    session = HTMLSession()
    r = session.get(product_url)

    zip_url_per_code = []
    for zipfile in r.html.find('a'):
        if ("zip" in zipfile.attrs['href']) and ("REFERENCE_MAP" not in zipfile.attrs['href']):
            zip_url_per_code.append("https://emergency.copernicus.eu"+zipfile.attrs['href'])
    
    return zip_url_per_code

