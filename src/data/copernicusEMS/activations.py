from requests_html import HTMLSession
import pandas as pd
import requests
import os
import shutil
from typing import List
import zipfile
from glob import glob
from shapely.ops import cascaded_union
import numpy as np
import datetime
import geopandas as gpd
import json
from src.data import utils


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
    # tables_floods["CodeDate"] = tables_floods["CodeDate"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))

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
        print("\tFile %s exists will not be download"%url_zip_file)
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


def generate_floodmap(register, filename_floodmap, filterland=True):
    """ Generates a floodmap (shapefile) with the joined info of the hydro and flood content. """

    area_of_interest = gpd.read_file(register["area_of_interest_file"])
    area_of_interest_pol = cascaded_union(area_of_interest["geometry"]) # register["area_of_interest"]

    mapdf = utils.filter_pols(gpd.read_file(register["observed_event_file"]),
                              area_of_interest_pol)
    assert mapdf.shape[0] > 0, f"No polygons within bounds for {register}"

    floodmap = gpd.GeoDataFrame({"geometry": mapdf.geometry},
                                crs=mapdf.crs)
    floodmap["w_class"] = mapdf["notation"]
    floodmap["source"] = "flood"

    if "hydrology_file" in register:
        mapdf_hydro = utils.filter_pols(gpd.read_file(register["hydrology_file"]),
                                        area_of_interest_pol)
        mapdf_hydro = utils.filter_land(mapdf_hydro) if filterland and (mapdf_hydro.shape[0] > 0) else mapdf_hydro
        if mapdf_hydro.shape[0] > 0:
            mapdf_hydro["source"] = "hydro"
            mapdf_hydro = mapdf_hydro.rename({"obj_type": "w_class"}, axis=1)
            mapdf_hydro = mapdf_hydro[["geometry", "w_class", "source"]].copy()
            floodmap = pd.concat([floodmap, mapdf_hydro], axis=0, ignore_index=True)

    # TODO add "hydrology_file_l"?? must be handled in later in create_gt.compute_water function

    # Concat area of interest
    area_of_interest["source"] = "area_of_interest"
    area_of_interest["w_class"] = "area_of_interest"
    area_of_interest = area_of_interest[["geometry", "w_class", "source"]].copy()
    floodmap = pd.concat([floodmap, area_of_interest], axis=0, ignore_index=True)

    floodmap.loc[floodmap.w_class.isna(), 'w_class'] = "Not Applicable"

    if filename_floodmap is not None:
        floodmap.to_file(filename_floodmap, driver="GeoJSON")

    return floodmap


def filter_register_copernicusems(unzipped_directory, code_date, verbose=False):
    """
    Collects source, observed event, hydro and area of interest shapefiles from a directory (unzipped)

    Args:
        unzipped_directory:
        code_date:
        verbose:

    Returns:

    """
    source_files = glob(os.path.join(unzipped_directory, "*_source*.dbf"))
    if len(source_files) != 1:
        print(f"Source file not found in directory {unzipped_directory}")
        return
    source_file = source_files[0]

    area_of_interest_files = glob(os.path.join(unzipped_directory, "*_observed*.shp"))
    if len(area_of_interest_files) != 1:
        print(f"Observed event file not found in directory {unzipped_directory}")
        return
    observed_event_file = area_of_interest_files[0]

    area_of_interest_files = glob(os.path.join(unzipped_directory, "*_area*.shp"))
    if len(area_of_interest_files) != 1:
        print(f"Area of interest file not found in directory {unzipped_directory}")
        return

    area_of_interest_file = area_of_interest_files[0]

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

    diff_dates = date_post_event - date_ems_code

    if diff_dates.days < 0:
        if verbose:
            print("Different between dates is negative %d" % diff_dates.days)
        return

    if diff_dates.days > 35:
        if verbose:
            print("difference too big %d" % diff_dates.days)
        return

    if (max_date_post_event-date_post_event).days >= 10:
        if verbose:
            print("difference between max date post event and min date post event too big %d" % (max_date_post_event-date_post_event).days)
        return

    stuff_pre_event = {}
    if np.any(pd_source.eventphase == "Pre-event"):
        date_pre_event = max(np.array(pd_source[pd_source.eventphase == "Pre-event"]["date"]))
        satellite_pre_event = np.array(pd_source[(pd_source.eventphase == "Pre-event") &
                                                 (pd_source.date == date_pre_event)]["source_nam"])[0]

        stuff_pre_event["satellite_pre_event"] = satellite_pre_event
        stuff_pre_event["timestamp_pre_event"] = date_pre_event
        if (date_post_event - date_pre_event).days < 0:
            if verbose:
                print("Date pre event %s is after date post event %s" % (date_pre_event.strftime("%Y-%m-%d"),
                                                                         date_post_event.strftime("%Y-%m-%d")))
            return

    # Filter content of shapefile
    pd_geo = gpd.read_file(observed_event_file)
    if np.any(pd_geo["event_type"] != '5-Flood'):
        if verbose:
            print("%s Event type is not Flood"%observed_event_file)
        return

    if not isinstance(satellite_post_event, str):
        if verbose:
            print("Satellite post event: {} is not a string!".format(satellite_post_event))
        return

    area_of_interest = gpd.read_file(area_of_interest_file)
    area_of_interest_pol = cascaded_union(area_of_interest["geometry"])

    register = {"name": product_name,
                "ems_code": ems_code,
                "timestamp": date_post_event,
                "satellite": satellite_post_event,
                "area_of_interest" : area_of_interest_pol,
                "timestamp_ems_code": date_ems_code,
                "observed_event_file": observed_event_file,
                "area_of_interest_file": area_of_interest_file}

    register.update(stuff_pre_event)

    hidrology_files = glob(os.path.join(unzipped_directory, "*_hydrography*.shp"))
    if len(hidrology_files) == 0:
        return register

    name_possibilities = ["_hydrographyA_", "_hydrography_a"]
    for name_pos in name_possibilities:
        hydrology_file_as = glob(os.path.join(unzipped_directory, f"*{name_pos}*.shp"))
        if len(hydrology_file_as) == 1:
            register["hydrology_file"] = hydrology_file_as[0]

    name_possibilities = ["_hydrographyL_", "_hydrography_l"]
    for name_pos in name_possibilities:
        hydrology_file_l = glob(os.path.join(unzipped_directory, f"*{name_pos}*.shp"))
        if len(hydrology_file_l) == 1:
            register["hydrology_file_l"] = hydrology_file_l[0]

    return register

def get_bbox(pd_geo):
    bounds = pd_geo.bounds
    minx = np.min(bounds.minx)
    miny = np.min(bounds.miny)
    maxx = np.max(bounds.maxx)
    maxy = np.max(bounds.maxy)

    return {"west": minx, "east": maxx, "north": maxy, "south": miny}


COPY_FORMATS = ["dbf", "prj", "shp", "shx"]

def processing_register(register, folder, bucket):

    filename = register["observed_event_file"]

    # bucket path (not local)
    layer_name = os.path.join("worldfloods/maps/", folder, os.path.splitext(os.path.basename(filename.replace(" ", "_")))[0])

    files_to_copy = [f"{os.path.splitext(filename)[0]}.{ext}" for ext in COPY_FORMATS]
    copy_to_bucket(bucket, files_to_copy, layer_name)

    pd_geo = gpd.read_file(filename)
    bounding_box = get_bbox(pd_geo)

    if "obj_desc" in pd_geo:
        event_type = np.unique(pd_geo.obj_desc)
        if len(event_type) > 1:
            print("Multiple event types within shapefile {}".format(event_type))
        event_type = event_type[0]
    else:
        event_type = "NaN"

    crs_code_space, crs_code = pd_geo.crs["init"].split(":")
    meta = {
        'event id': register["name"],
        'layer name': os.path.basename(layer_name),
        'event type': event_type,
        'satellite date': register["timestamp"].isoformat(),
        'country': "NaN",
        'satellite': register["satellite"],
        'bounding box': bounding_box,
        'reference system': {
            'code space': crs_code_space,
            'code': crs_code
        },
        'abstract': "NaN",
        'purpose': "NaN",
        'source': 'CopernicusEMS'
    }
    blob_name = os.path.join(layer_name, "meta.json")
    if bucket.get_blob(blob_name) is None:
        with open("meta.json", "w") as fh:
            json.dump(meta, fh)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename("meta.json")


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
    elif len(os.listdir(directory_to_extract_to)) > 0:
        return directory_to_extract_to

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
    """
    https://emergency.copernicus.eu/mapping/list-of-components/EMSR502/aemfeed
    """
    product_url = "https://emergency.copernicus.eu/mapping/list-of-components/" + code
    session = HTMLSession()
    r = session.get(product_url)

    zip_url_per_code = []
    for zipfile in r.html.find('a'):
        if ("zip" in zipfile.attrs['href']) and ("REFERENCE_MAP" not in zipfile.attrs['href']):
            zip_url_per_code.append("https://emergency.copernicus.eu"+zipfile.attrs['href'])
    
    return zip_url_per_code

