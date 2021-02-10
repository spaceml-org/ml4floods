import json
import os
from datetime import datetime
from tqdm import tqdm
from shapely.geometry import Polygon
import re
from utils import files_errors_train_test as ftt
import utils.visutils
import geopandas as gpd
import io
import subprocess
from glob import glob


RENAME_SATELLITE = {"landsat 5": "Landsat-5",
                    "landsat 8": "Landsat-8",
                    "landsat-8": "Landsat-8",
                    "landsat 7": "Landsat-7",
                    "sentinel-1": "Sentinel-1",
                    "pleadies": "Pleiades-1A-1B",
                    "sentinel-2": "Sentinel-2",
                    "radarsat-1": "RADARSAT-1",
                    "radarsat-2": "RADARSAT-2",
                    "terrasar-x": "TERRASAR-X",
                    'spot 6' : "SPOT-6-7",
                    "worldview-2": "WorldView-2"}

SATELLITE_TYPE = {'COSMO-SkyMed': "SAR",
                  'GeoEye-1': "SAR",
                  'Landsat-5': "Optical",
                  'Landsat-7': "Optical",
                  'Landsat-8': "Optical",
                  'PlanetScope': "Optical",
                  'Pleiades-1A-1B': "Optical",
                  'RADARSAT-1': "SAR",
                  'RADARSAT-2': "SAR",
                  'SPOT-6-7': "Optical",
                  'Sentinel-1': "SAR",
                  'Sentinel-2': "Optical",
                  'TERRASAR-X': "SAR",
                  'WorldView-1': "Optical",
                  'WorldView-2': "Optical",
                  'WorldView-3': "Optical",
                  'alos palsar': "SAR",
                  'asar imp': "SAR",
                  'dmc': "Optical",
                  'earth observing 1': "Optical",
                  'modis-aqua': "Optical",
                  'modis-terra': "Optical",
                  'spot4': "Optical"}


def bbox_2_pol(bbox, shapelypolygon=False):
    """
    Generates a list of coordinates: [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x1,y1]]
    """

    pol_list = [[bbox[0], bbox[1]],
                [bbox[2], bbox[1]],
                [bbox[2], bbox[3]],
                [bbox[0], bbox[3]],
                [bbox[0], bbox[1]]]

    if shapelypolygon:
        pol_list = Polygon(pol_list)
    return pol_list


def load_map(register, worldfloods_root="gs://worldfloods"):
    """
    Load the map.shp corresponding to the given register as geopandas dataframe
    :param register:
    :param worldfloods_root: root worldfloods data
    :return:
    """
    resource_path = os.path.join(worldfloods_root, "maps/", register["resource folder"], register["layer name"], "map.shp")
    return gpd.read_file(resource_path)


def process_meta(filename):
    """ Process metadata from gs://worldfloods/maps folder """

    if filename.startswith("gs://"):
        from google.cloud import storage

        client = storage.Client()
        with io.BytesIO() as file_obj:
            client.download_blob_to_file(filename, file_obj)
            file_obj.seek(0)
            data = json.load(file_obj)
    else:
        with open(filename, "r") as fh:
            data = json.load(fh)

    basename_filename = os.path.basename(filename)  # meta.json
    folder_resource = os.path.dirname(filename)
    name_resource = os.path.basename(folder_resource)
    floodhydrofolder = os.path.dirname(folder_resource)

    if utils.exists_bucket_or_disk(os.path.join(floodhydrofolder+"_edited", name_resource, basename_filename)):
        data["resource folder"] = os.path.basename(floodhydrofolder+"_edited")
    else:
        data["resource folder"] = os.path.basename(floodhydrofolder)

    if data["satellite"] in RENAME_SATELLITE.keys():
        data["satellite"] = RENAME_SATELLITE[data["satellite"]]

    if data["satellite date"] is not None:
        data["satellite date"] = datetime.strptime(data["satellite date"], "%Y-%m-%dT%H:%M:%S")
    return data


def glob_gsutil(globloc, isdir=False):
    """" Glob using gsutils """

    if isdir:
        commands = ["gsutil", "ls", "-d", globloc]
    else:
        commands = ["gsutil", "ls", globloc]

    print(f"globbing: {globloc}")
    glob_result = subprocess.check_output(commands)
    return [g for g in glob_result.decode("UTF-8").split('\n') if g != ""]


def data_floods(include_all=False, exclude_errors_v2=False, worldfloods_root="gs://worldfloods"):
    """
    Return the processed meta.json files from folders extent and flood

    :param include_all: if False only events after the launch of Sentinel-2 will be included
    :param exclude_errors_v2: if True only layers not included in ftt.FILENAMES_ERRORS_V2 will be included
    :param worldfloods_root: Root of worldfloods data

    :return:
    """

    if worldfloods_root.startswith("gs://"):
        files_meta_flood = sorted(glob_gsutil(os.path.join(worldfloods_root, "maps/extent/*/meta.json")) + \
                                  glob_gsutil(os.path.join(worldfloods_root, "maps/flood/*/meta.json")))
    else:
        files_meta_flood = sorted(glob(os.path.join(worldfloods_root, "maps/extent/*/meta.json")) +\
                                  glob(os.path.join(worldfloods_root, "maps/flood/*/meta.json")))

    layers_errors = set()
    if exclude_errors_v2:
        for tiff in ftt.FILENAMES_ERRORS_V2:
            layers_errors.add(get_layer_name(tiff))

    dats = []
    for f in tqdm(files_meta_flood):
        data = process_meta(f)

        if data["satellite date"] is None:
            continue

        if not include_all and (data["satellite date"] < datetime.strptime("2015-07-01", "%Y-%m-%d")):
            continue

        if data["layer name"] in layers_errors:
            continue

        dats.append(data)

    return dats


def get_layer_name(tiffile):
    layer_name = os.path.splitext(os.path.basename(tiffile))[0]
    if re.search("\d{10}-\d{10}", layer_name) is not None:
        layer_name = layer_name[:-21]
    return layer_name


def location_CEMS(layer_name):
    return "_".join(layer_name.split("_")[:2])


def load_hydro_data(dats_floods, worldfloods_root="gs://worldfloods"):
    """ Add hydro metadata to dats_floods entries. Returns all hydros found in datas_floods """

    if worldfloods_root.startswith("gs://"):
        files_meta_hydro_waterways = sorted(glob_gsutil(os.path.join(worldfloods_root, "maps/hydrography/*/meta.json")) + \
                                            glob_gsutil(os.path.join(worldfloods_root, "maps/waterways/*/meta.json")))
    else:
        files_meta_hydro_waterways = sorted(glob(os.path.join(worldfloods_root, "maps/hydrography/*/meta.json")) + \
                                            glob(os.path.join(worldfloods_root, "maps/waterways/*/meta.json")))

    dats_extra = []
    for _i, f in enumerate(files_meta_hydro_waterways):
        data = process_meta(f)
        floods_related_hydro_extent = []
        for d in dats_floods:
            if d["event id"] == data["event id"]:
                assert "hydro" not in d, "hydro already in d"
                d["hydro"] = data
                floods_related_hydro_extent.append(d)

        if len(floods_related_hydro_extent) == 0:
            print("%d/%d %s does not have related flood" % (_i + 1, len(files_meta_hydro_waterways), f))
            continue

        data["floods related"] = floods_related_hydro_extent
        dats_extra.append(data)

    # Fill missing hydro with hydros of the same CopernicusEMS location if exists
    for d in dats_floods:
        if ("hydro" not in d) and (d["source"] == "CopernicusEMS"):
            for _i, data in enumerate(dats_extra):
                if (d["source"] == data["source"]) and \
                        (location_CEMS(d["layer name"]) == location_CEMS(data["layer name"])):
                    d["hydro"] = data
                    data["floods related"].append(d)
                    break

    return dats_extra

