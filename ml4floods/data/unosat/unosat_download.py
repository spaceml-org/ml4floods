import requests_html
import re
import time
from dataclasses import dataclass
import json
import geopandas
from fs import open_fs
import requests
import io
from zipfile import ZipFile
import shutil
import numpy as np
import traceback as tb
from ml4floods.data.unosat.unosat_download_arg_parser import UnosatDownloadArgParser

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
        
    def __hash__(self):
        return self.download_url.__hash__() + self.date_published.__hash__() + self.glide_id.__hash__()

def get_flood_shape_and_meta(session, download_base_regex, base_url, map_link):
    fetched_page = session.get(base_url + map_link)
    if has_shapefile(fetched_page):
        shape_link, glide_id = get_flood_shape_link(fetched_page, download_base_regex)
        if shape_link is not None:
            info = ShapeFileInfo()
            info.date_published = get_flood_shape_published_date(fetched_page)
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
    
    pd_geo = geopandas.read_file(shapefile_path)
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
        'layer name': os.path.basename(shapefile_info.download_url),
        'event type': event_type,
        'satellite date': shapefile_info.imagery_dates if shapefile_info.imagery_dates is not None else "NaN",
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
        contents = f.namelist()
        # TODO: Some zips only contain WaterExtent but no FloodExtent files -- we should investigate
        # what this means and whether we want to use these too (are they floods?)
        flood_extent_shapefiles = list(filter(lambda x: re.match(r".*\_FloodExtent\_.*\.shp$", x), contents))
        if len(flood_extent_shapefiles) > 0:
            if len(flood_extent_shapefiles) != 1:
                print("WARN: More than one flood shapefile... We will only be using the first one...")
            base_shape_name = os.path.splitext(flood_extent_shapefiles[0])[0]
            relevant_shapefiles = filter(lambda x: x.startswith(base_shape_name), contents)
            # We write all the relevant files out to disk like this because that's the only way I was
            # able to get GeoPandas to accept the file -- it seems not to like a BytesIO object (shapefile)
            # or a list of BytesIO objects (relevant files). It seems to need a path to a .shp file that is
            # located in the same place as its supporting files. If we want to optimize for efficiency,
            # this may be one place to look.
            for relevant_file in relevant_shapefiles:
                try:
                    with f.open(relevant_file, "r") as shp_f, open(f"./tmp/{relevant_file}", "wb") as f_out:
                        f_out.write(shp_f.read())
                except NotImplementedError:
                    print("Unimplemented compression... Skipping...")
                    tb.print_exc()
                    return None
                    

            return "./tmp/" + flood_extent_shapefiles[0]
        return None


def download_shapefiles(shapefile_infos, shapefile_output_dir, metadata_output_dir):
    
    for shapefile_info in shapefile_infos:
        print(shapefile_info)
        shapefiles_and_dbf_zip = requests.get(shapefile_info.download_url).content
        shapefile_path = get_shapefile_from_zip(io.BytesIO(shapefiles_and_dbf_zip))
        if shapefile_path != None:
            meta = produce_metadata_dict(shapefile_info, shapefile_path)
            metadata_name = os.path.splitext(meta["layer name"])[0] + ".json"
            with open_fs(metadata_output_dir +"?strict=False") as fs:
                with fs.open(metadata_name, "w") as f:
                    json.dump(meta, f)

            shapefile_name = shapefile_path.replace("./tmp/", "")
            shapefile_basename = os.path.splitext(shapefile_name)[0]
            layer_basename = os.path.splitext(meta["layer name"])[0]
            for tmpfile in os.listdir("./tmp"):
                target_fname = tmpfile.replace(shapefile_basename, layer_basename)
                with open_fs(os.path.join(shapefile_output_dir) + "?strict=False") as fs:
                    with fs.open(target_fname, "wb") as f, open("./tmp/"+tmpfile, "rb") as f_in:
                        f.write(f_in.read())

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

