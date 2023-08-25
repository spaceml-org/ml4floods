import requests_html
import re
import time
from dataclasses import dataclass
import json
import geopandas as gpd
from fs import open_fs
import requests
import io
from zipfile import ZipFile
import shutil
import numpy as np
import traceback as tb
from ml4floods.data.unosat.unosat_download_arg_parser import UnosatDownloadArgParser
from ml4floods.data import utils
import os
from typing import Dict


RAW_ZIP_PATH = "gs://ml4cc_data_lake/0_DEV/0_Raw/WorldFloods/shapefiles/unosat/"
RAW_META_PATH = "gs://ml4cc_data_lake/0_DEV/0_Raw/WorldFloods/meta/"
STATING_PATH = "gs://ml4cc_data_lake/0_DEV/1_Staging/WorldFloods/"

def get_map_links(fetched_page):
    map_links = list(filter(lambda x: re.match(r"^/maps/map/[0-9]+", x), fetched_page.html.links))

    next_page_links = list(filter(lambda x: re.match(r"\?page=[0-9]+", x), fetched_page.html.links))
    next_page_nums = set([int(re.match(r"\?page=([0-9]+)", x).group(1)) for x in next_page_links])
    
    return map_links, next_page_nums
    
def has_shapefile(fetched_page):
    """
    Returns True if a UNITAR product page like https://unitar.org/maps/map/3206 has a shapefile
    """
    return "SHAPEFILE" in fetched_page.content.decode(fetched_page.encoding).upper()
    
def get_flood_shape_link(fetched_page, download_base):
    glide_ids = re.findall(r"GLIDE\:( |&nbsp;)(FL[0-9]+[A-Z]+)", fetched_page.content.decode(fetched_page.encoding))
    if len(glide_ids) > 0:
        # This means we have a GLIDE ID
        assert len(glide_ids) == 1
        shape_links = list(filter(lambda x: re.match(f"{download_base}[a-zA-Z0-9\-/]+{glide_ids[0][1]}_(SHP|shp)\.zip", x), fetched_page.html.links))
        if len(shape_links) > 0:
            assert len(shape_links) == 1
            glide_id = glide_ids[0][1]
            return shape_links[0], glide_id
    
    return None, None
    
def get_country_links_from_url(session, country_list_url):
    fetched_page = session.get(country_list_url)
    return list(filter(lambda x: re.match(r"^/maps/countries/[0-9]+", x), fetched_page.html.links))

def get_map_links_from_country_links_and_url(session, base_url, country_links):
    all_map_links = []
    for country_link in country_links:
        next_page_nums = set([0])
        cur_page = -1
        while cur_page + 1 in next_page_nums:
            cur_page = cur_page + 1

            fetched_page = session.get(base_url + country_link + f"?page={cur_page}")
            map_links, next_page_nums = get_map_links(fetched_page)
            all_map_links.extend(map_links)
            time.sleep(0.2)
    return all_map_links

def get_flood_shape_published_date(fetched_page):
    published_dates = re.findall(r"Published\:( |&nbsp;)([0-9A-Za-z ,]+)", fetched_page.content.decode(fetched_page.encoding))
    if len(published_dates) > 0:
        # This means we have a published date
        assert len(published_dates) == 1
        return published_dates[0][1]
    return None

def get_floodmap_resolution(fetched_page):
    resolution = re.findall(r"Resolution\:( |&nbsp;)([0-9A-Za-z ,]+)", fetched_page.content.decode(fetched_page.encoding))
    if len(resolution) > 0:
        # This means we have a resolution
        # assert len(resolution) == 1
        return int(resolution[0][1][:-2])
    return None

def get_satellite_source(fetched_page):
    matches = re.findall(r"Satellite Data( |&nbsp;)*(\((Post|Pre|[0-9]+)\))?( |&nbsp;)*\:( |&nbsp;)*([0-9A-Za-z ,\-]+)", fetched_page.content.decode(fetched_page.encoding))
    if len(matches) > 0:
        return matches[0][5]
    return None

def get_satellite_imagery_dates(fetched_page):
    matches = re.findall(r"Imagery Dates( |&nbsp;)*\:( |&nbsp;)*([0-9A-Za-z ,]+)", fetched_page.content.decode(fetched_page.encoding))
    if len(matches) > 0:
        # Sometimes, we see multiple "Imagery Dates" sections (in 50/245 floods as of 02/28/2021)
        # Example: https://unitar.org/maps/map/2562
        return matches[0][2]
    return None

@dataclass
class ShapeFileInfo:
    """Download link and metadata for a shapefile"""
    download_url: str = None
    date_published: str = None
    source_url: str = None
    glide_id: str = None
    satellite_data: str = None
    imagery_dates: str = None
    resolution: int = None
        
    def __hash__(self):
        return self.download_url.__hash__() + self.date_published.__hash__() + self.glide_id.__hash__()

# def get_flood_shape_and_meta(session, download_base_regex, base_url, map_link):
#     fetched_page = session.get(base_url + map_link)
#     if has_shapefile(fetched_page):
#         shape_link, glide_id = get_flood_shape_link(fetched_page, download_base_regex)
#         if shape_link is not None:
#             info = ShapeFileInfo()
#             info.date_published = get_flood_shape_published_date(fetched_page)
#             info.glide_id = glide_id
#             info.download_url = shape_link
#             info.source_url = base_url + map_link
#             # Get satellite source -- this is formatted in many different ways
#             # and needs more work, so will not always be populated. In instances
#             # with separate pre-event and post-event imagery, only the first
#             # mentioned on the page will be taken.
#             info.satellite_data = get_satellite_source(fetched_page)
#             # Get imagery dates -- this is not always populated, like satellite data
#             # There are many different ways UNOSAT expresses dates over the pages.
#             # More work would be helpful here in the long term
#             info.imagery_dates = get_satellite_imagery_dates(fetched_page)
            
#             return info
#     return None

def get_map_link(session, map_url):
    map_links = session.get(map_url).html.absolute_links
    vector_url = list(filter(lambda x: re.match(r'https://unosat-maps\.web\.cern\.ch/[^/]+/[^/]+/[^/]+_SHP\.zip', x), map_links))
    return vector_url

def get_all_map_links(session, base_url, only_first_page = True):

    all_links = []
    first_page_url = 'https://unitar.org/maps/all-maps?page=0'
    # first_page_url = base_url + '?page=0'
    first_page_links = session.get(first_page_url).html.absolute_links

    pages_urls = list(filter(lambda x: re.match(r'.*\?page=.*', x), first_page_links))
    last_page = max([int(f.split('=')[-1]) for f in pages_urls])
    for page_number in range(last_page):
        
        if page_number > 0 and only_first_page:
            break
        page_url = f'https://unitar.org/maps/all-maps?page={page_number}'
        page = session.get(page_url)
        all_links.extend(page.html.absolute_links)
        print(f'Finished page {page_number}')

        
    map_links = list(filter(lambda x: re.match(r'.*\/maps/map/.*', x), all_links))
    
    # vector_links = []
    # for map_link in map_links[0:2]:
    #     vector_links.extend(get_map_link(session, map_link))

    return map_links

def get_flood_shape_and_meta(session, download_base_regex, map_link):
    base_url = "https://unitar.org"
    fetched_page = session.get(map_link)
    if has_shapefile(fetched_page):
        shape_link, glide_id = get_flood_shape_link(fetched_page, download_base_regex)
        if shape_link is not None:
            info = ShapeFileInfo()
            info.date_published = get_flood_shape_published_date(fetched_page)
            info.resolution = get_floodmap_resolution(fetched_page)
            info.glide_id = glide_id
            info.download_url = shape_link
            info.source_url = base_url + map_link
            # Get satellite source -- this is formatted in many different ways
            # and needs more work, so will not always be populated. In instances
            # with separate pre-event and post-event imagery, only the first
            # mentioned on the page will be taken.
            info.satellite_data = get_satellite_source(fetched_page)
            # Get imagery dates -- this is not always populated, like satellite data
            # There are many different ways UNOSAT expresses dates over the pages.
            # More work would be helpful here in the long term
            info.imagery_dates = get_satellite_imagery_dates(fetched_page)
            
            return info
    return None

def get_flood_shapefiles(session, base_url, country_list_url, download_base_regex):
    all_country_links = get_country_links_from_url(session, country_list_url)
    all_map_links = get_map_links_from_country_links_and_url(session, base_url, all_country_links)
    flood_shapes = []
    for link in all_map_links:
        fetched_page = session.get(base_url + link)
        info = get_flood_shape_and_meta(session, download_base_regex, base_url, link)
        if info is not None:
            flood_shapes.append(info)
        time.sleep(0.08)
    return flood_shapes


def get_bbox(pd_geo):
    bounds = pd_geo.bounds
    minx = np.min(bounds.minx)
    miny = np.min(bounds.miny)
    maxx = np.max(bounds.maxx)
    maxy = np.max(bounds.maxy)

    return {"west": minx, "east": maxx, "north": maxy, "south": miny}

def produce_metadata_dict(shapefile_info, shapefile_path):
    
    pd_geo = gpd.read_file(shapefile_path)
    bounding_box = get_bbox(pd_geo)

    if "obj_desc" in pd_geo:
        event_type = np.unique(pd_geo.obj_desc)
        if len(event_type) > 1:
            print("Multiple event types within shapefile {}".format(event_type))
        event_type = event_type[0]
    else:
        event_type = "NaN"
        
    crs_code_space, crs_code = str(pd_geo.crs).split(":")
        
    meta = {
        'event id': shapefile_info.glide_id,
        'ems_code': f"UNOSAT{shapefile_info.glide_id}",
        'aoi_code': shapefile_path.split("_FloodExtent_")[-1].split(".shp")[0],
        'layer name': os.path.basename(shapefile_info.download_url),
        'event type': event_type,
        'satellite date': shapefile_info.imagery_dates if shapefile_info.imagery_dates is not None else "NaN",
        'date_ems_code': shapefile_info.date_published if shapefile_info.date_published is not None else "NaN",
        'country': "NaN",
        'satellite': shapefile_info.satellite_data if shapefile_info.satellite_data is not None else "NaN",
        'bounding box': bounding_box,
        'reference system': {
            'code space': crs_code_space,
            'code': crs_code
        },
        'abstract': "NaN",
        'purpose': "NaN",
        'source': 'UNOSAT'
    }
    
    return meta

def get_shapefile_from_zip(zip_file):
    
    try:
        shutil.rmtree("./tmp")
    except:
        pass    
    os.mkdir("./tmp")
    
    with ZipFile(zip_file, "r") as f:
        f.extractall('./tmp/')
        contents = f.namelist()
        # TODO: Some zips only contain WaterExtent but no FloodExtent files -- we should investigate
        # what this means and whether we want to use these too (are they floods?)
        flood_extent_shapefiles = list(filter(lambda x: re.match(r".*\_FloodExtent\_.*\.shp$", x), contents))
        if len(flood_extent_shapefiles) > 0:
            if len(flood_extent_shapefiles) != 1:
                print("WARN: More than one flood shapefile... We will only be using the first one...")
            base_shape_name = os.path.splitext(flood_extent_shapefiles[0])[0]
            relevant_shapefiles = list(filter(lambda x: x.startswith(base_shape_name), contents))

            # We write all the relevant files out to disk like this because that's the only way I was
            # able to get GeoPandas to accept the file -- it seems not to like a BytesIO object (shapefile)
            # or a list of BytesIO objects (relevant files). It seems to need a path to a .shp file that is
            # located in the same place as its supporting files. If we want to optimize for efficiency,
            # this may be one place to look.
            
            
            # for relevant_file in relevant_shapefiles:
            #     os.makedirs(os.path.dirname(f"./tmp/{relevant_file}"), exist_ok=True)
            #     try:
            #         with f.open(relevant_file, "r") as shp_f, open(f"./tmp/{relevant_file}", "wb") as f_out:
            #             f_out.write(shp_f.read())
            #     except NotImplementedError:
            #         print("Unimplemented compression... Skipping...")
            #         tb.print_exc()
            #         return None
                    

            return "./tmp/" + flood_extent_shapefiles[0]
        return None
    
def update_metadata_dict(meta: Dict, flood_source: gpd.GeoDataFrame, area_of_interest: gpd.GeoDataFrame):
    
    flood_info = flood_source.to_dict()
    meta['area_of_interest_polygon'] = area_of_interest.geometry.unary_union
    meta['satellite date'] = datetime.strptime(flood_info['Sensor_Dat'][0], "%Y-%m-%d")
    meta['date_ems_code'] = meta['satellite date']
    meta['confidence'] = flood_info['Confidence'][0]
    meta['water_status'] = flood_info['Water_Stat'][0]
    
    return meta

def read_unosat_shapefile(folder: str, shp_name: str, class_dict: Dict):

        source = gpd.read_file(f"{folder}/{shp_name[0]}")
        pols = source.geometry.explode().reset_index(drop=True).to_frame()
        pols['source'] = class_dict['source']
        pols['w_class'] = class_dict['w_class']
        
        return pols
        
def extract_unosat_staging(unzipped_path:str, metadata_dict: Dict):
    """
    
    
    """

    files_unziped = os.listdir(unzipped_path)

    # get the first flood mask file, which probably contains the peak flood information
    flood_files = list(filter(lambda x: re.match(r".*\_FloodExtent\_.*\.shp$", x), files_unziped))
    pattern_first = flood_files[0].split('_FloodExtent')[0] 
    area_of_interest_file = list(filter(lambda x: re.match(rf".*\{pattern_first}_AnalysisExtent\_.*\.shp$", x), files_unziped))
    water_extent_file = list(filter(lambda x: re.match(rf".*\{pattern_first}_WaterExtent\_.*\.shp$", x), files_unziped))
    permanent_water_file = list(filter(lambda x: re.match(rf".*\{pattern_first}_PermanentWater\_.*\.shp$", x), files_unziped))


    flood_pols = read_unosat_shapefile(unzipped_path, flood_files, class_dict = {'source': 'flood', 'w_class': 'Flooded area'})
    if len(water_extent_file) > 0:
        water_pols = read_unosat_shapefile(unzipped_path, water_extent_file, class_dict = {'source': 'flood', 'w_class': 'Flooded area'})
    if len(permanent_water_file) > 0:
        permanent_pols = read_unosat_shapefile(unzipped_path, permanent_water_file, class_dict = {'source': 'hydro', 'w_class': 'Not Applicable'})

    floodmap = pd.concat([flood_pols, water_pols, permanent_pols], axis = 0)
    floodmap = floodmap.dissolve(by='source')
    floodmap = floodmap.explode().reset_index(drop=True)
    floodmap['source'] = floodmap.w_class.apply(lambda x: 'flood' if x == 'Flooded area' else 'hydro')
    # Maybe eliminate small polygons

    if len(area_of_interest_file) > 0:
        area_of_interest = gpd.read_file(f"{unzipped_path}/{area_of_interest_file[0]}")
        floodmap = floodmap.clip(area_of_interest)
    # else:
    #     area_of_interest = gpd.GeoDataFrame(geometry=[box(*floodmap.total_bounds)], crs=floodmap.crs)
    #     area_of_interest['source'] = 'area_of_interest'
    #     area_of_interest['w_class'] = 'area_of_interest'

    floodmap = pd.concat([floodmap, area_of_interest], axis = 0)
    
    ## Update metadata
    
    flood_source = gpd.read_file(f"{unzipped_path}/{flood_files[0]}")
    metadata_dict = update_metadata_dict(metadata_dict, flood_source, area_of_interest)
    
    return floodmap, metadata_dict


def download_shapefiles(shapefile_infos, shapefile_output_dir, metadata_output_dir):
    
    fs_bucket = utils.get_filesystem(RAW_ZIP_PATH)
    
    for i, shapefile_info in enumerate(shapefile_infos):
        print(f"Processing shapefile {i+1}/{len(shapefile_infos)}: {shapefile_info.glide_id} ")
        if shapefile_info.resolution > 50:
            print("Skipping shapefile with resolution > 50m")
            continue
        try:
            shapefiles_and_dbf_zip = requests.get(shapefile_info.download_url).content
            shapefile_path = get_shapefile_from_zip(io.BytesIO(shapefiles_and_dbf_zip))
            if shapefile_path != None:
                meta = produce_metadata_dict(shapefile_info, shapefile_path)
                # metadata_name = os.path.splitext(meta["layer name"])[0] + ".json"
                
                # Save raw zip to gcp
                zipfile_bucket_path = RAW_ZIP_PATH + f"{meta['event id']}_{meta['aoi_code']}.zip"
                
                with open(f"./tmp/{meta['event id']}", "wb") as f:
                    f.write(shapefiles_and_dbf_zip)
                fs_bucket.put_file(f"./tmp/{meta['event id']}",zipfile_bucket_path)
                
                # Write metadata to gcp
                meta_bucket_path = metadata_output_dir + f"{meta['event id']}_{meta['aoi_code']}.json"
                utils.write_json_to_gcp(meta_bucket_path, meta)
                
                ## Function to process floodmap and write to Staging
                
                print('Exrtacting Staging data')
                floodmap, meta = extract_unosat_staging(os.path.dirname(shapefile_path), meta)
                
                flood_date = meta['satellite date'].strftime("%Y%m%d")
                staging_floodmap_path = os.path.join(STATING_PATH, meta['ems_code'], meta['aoi_code'], 'floodmap',f'{flood_date}.geojson')
                utils.write_geojson_to_gcp(staging_floodmap_path, floodmap)
                staging_pickle_path = os.path.join(STATING_PATH, meta['ems_code'], meta['aoi_code'], 'flood_meta',f'{flood_date}.pickle')
                utils.write_pickle_to_gcp(staging_pickle_path, meta)
                
                # shapefile_name = shapefile_path.replace("./tmp/", "")
                # shapefile_basename = os.path.splitext(shapefile_name)[0]
                # layer_basename = os.path.splitext(meta["layer name"])[0]
                # for tmpfile in os.listdir("./tmp"):
                #     target_fname = tmpfile.replace(shapefile_basename, layer_basename)
                #     with open_fs(os.path.join(shapefile_output_dir) + "?strict=False") as fs:
                #         with fs.open(target_fname, "wb") as f, open("./tmp/"+tmpfile, "rb") as f_in:
                #             f.write(f_in.read())
        except Exception as e:
            print(f"Error processing shapefile {shapefile_info.glid_id} -- skipping...")
            tb.print_exc()
            continue
        

def run_unosat_pipeline(base_url, country_list_url, download_base_regex, shapefile_bucket, metadata_bucket):
    session = requests_html.HTMLSession()
    flood_shapefiles = get_flood_shapefiles(session, base_url, country_list_url, download_base_regex)
    download_shapefiles(flood_shapefiles, shapefile_bucket, metadata_bucket)

if __name__ == "__main__":
    args = UnosatDownloadArgParser().parse_args()
    base_url = args.base_url
    country_list_url = args.country_list_url
    download_base_regex = args.download_base_regex
    shapefile_bucket = args.shapefile_bucket
    metadata_bucket = args.metadata_bucket
    run_unosat_pipeline(base_url, country_list_url, download_base_regex, shapefile_bucket, metadata_bucket)

