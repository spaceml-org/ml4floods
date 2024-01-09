import traceback
import warnings

import ee
import time
import os
from glob import glob
from typing import Optional, Callable, List, Tuple, Union
from shapely.geometry import mapping, Polygon
import numpy as np
import geopandas as gpd
import pandas as pd
from ml4floods.data.utils import get_filesystem
from datetime import datetime, timezone
import math

BANDS_NAMES = {
    # Sentinel-2 L1C COPERNICUS/S2 COPERNICUS/S2_HARMONIZED
    "COPERNICUS/S2_HARMONIZED" : ["B1","B2","B3","B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12", "QA60"],
    # Sentinel-2 L2A COPERNICUS/S2_SR
    "COPERNICUS/S2_SR_HARMONIZED" : ["B1","B2","B3","B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "SCL"],
    # Landsat-8/9 TOA
    "Landsat" : ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11", "QA_PIXEL"]
}


def _get_collection(collection_name, date_start, date_end, bounds):
    collection = ee.ImageCollection(collection_name)
    collection_filtered = collection.filterDate(date_start, date_end) \
        .filterBounds(bounds)

    n_images = int(collection_filtered.size().getInfo())

    return collection_filtered, n_images

    # add cloud probability
def add_cloud_prob_landsat(img:ee.Image) -> ee.Image:
    bqa = img.select(["QA_PIXEL"], ["probability"])
    clouds = bqa.bitwise_and(int("0000000000001000",2)).gt(0).multiply(100).toUint16()

    # Store images in S2 range
    img_radiances = img.select(BANDS_NAMES["Landsat"][:-1]).multiply(10_000).toUint16()
    img_return = img_radiances.addBands(img.select("QA_PIXEL")).addBands(clouds)
    img_return = img_return.copyProperties(img)
    img_return = img_return.set("system:time_start", img.get("system:time_start"))
    return img_return

def get_landsat_collection(date_start:datetime, date_end:datetime,
                           bounds:ee.Geometry,
                           verbose:int=1) -> Optional[ee.ImageCollection]:
    """
    Returns a Landsat-8 and Landsat-9 time series of daily mosaicked images over the bounds. It adds a probability band
    to mimic the S2 case; additionally values are 10_000 and casted to uint16.
    (also to maintain the similarity with S2 collections).

    Args:
        date_start: start search date
        date_end: end search date
        bounds: polygon with the AoI to download
        verbose: print stuff

    Returns:
        ImageCollection

    """
    # GEE doesnt like time zones
    date_start = date_start.replace(tzinfo=None)
    date_end = date_end.replace(tzinfo=None)
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_RT_TOA").filterDate(date_start, date_end).filterBounds(bounds)
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_TOA").filterDate(date_start, date_end).filterBounds(bounds)
    l89 = l8.merge(l9)
    n_images = int(l89.size().getInfo())
    if n_images <= 0:
        if verbose > 1:
            print(f"Not images found for collection LANDSAT/LC08/C02/T1_RT_TOA and LANDSAT/LC09/C02/T1_TOA date start: {date_start} date end: {date_end}")
        return

    l89 = l89.map(add_cloud_prob_landsat)

    daily_mosaic = collection_mosaic_day(l89, bounds)
    # fun_before_mosaic=lambda img: img.toFloat().resample("bicubic")) # Bicubic resampling for 60m res bands?

    # Add cloud info and valid info
    count_fun = get_count_function(["QA_PIXEL"], bounds)
    daily_mosaic = daily_mosaic.map(count_fun)

    return daily_mosaic

def add_probability_from_BQA(img:ee.Image) -> ee.Image:
    """
    Adds to the image the cloud probability in [0,100] computed from BQA layer

    # Code taken from https://github.com/IPL-UV/ee_ipl_uv

    var qa = image.select('QA60');
    // Bits 10 and 11 are clouds and cirrus, respectively.
    var cloudBitMask = 1 << 10;
    var cirrusBitMask = 1 << 11;

    // Both flags should be set to zero, indicating clear conditions.
    var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
                 .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
    """
    qa = img.select(['QA60'], ["probability"])
    cloud_bit_mask =  1 << 10 # int("0000010000000000", 2)
    cirrus_bit_mask = int("0000100000000000", 2)
    # qa.bitwiseAnd(cloud_bit_mask).gt(0).Or(qa.bitwiseAnd(cirrus_bit_mask).gt(0))

    # Ignore cirrus since we can predict on water in those areas
    cloud_mask = qa.bitwiseAnd(cloud_bit_mask).gt(0).multiply(100).toUint16()
    return img.addBands(cloud_mask)


def get_s2_collection(date_start:datetime, date_end:datetime,
                      bounds:ee.Geometry,
                      collection_name:str="COPERNICUS/S2_HARMONIZED", bands:Optional[List[str]]=None,
                      force_s2cloudless:bool=True,
                      verbose:int=1) -> Optional[ee.ImageCollection]:
    """
    Returns an ee.ImageCollection with mosaicked S2 images joined with the s2cloudless cloud masks
    (see collection COPERNICUS/S2_CLOUD_PROBABILITY). This function also filters repeated images of the same date
    over the same location.

    Args:
        date_start: start search date
        date_end: end search date
        bounds: polygon with the AoI to download
        collection_name: "COPERNICUS/S2" for L1C Sentinel-2 images and ""COPERNICUS/S2_SR" for L2A images.
        bands: list of bands to get
        force_s2cloudless: if s2cloudless is missing for some S2 images it will skip those instead of raising
        not implemented errors.
        verbose: print stuff

    Returns:
        image collection with S2 imaages and s2cloudless cloud masks.
        None if no image satisfies the condition in the collection
    """

    # GEE doesnt like time zones
    date_start = date_start.replace(tzinfo=None)
    date_end = date_end.replace(tzinfo=None)
    img_col_all, n_images_col = _get_collection(collection_name, date_start, date_end, bounds)
    if n_images_col <= 0:
        if verbose > 1:
            print(f"Not images found for collection {collection_name} date start: {date_start} date end: {date_end}")
        return

    if bands is None:
        bands = BANDS_NAMES[collection_name]

    img_col_all = img_col_all.select(bands)

    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        # .filterBounds(bounds)
        .filterDate(date_start, date_end))

    img_col_all_join = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': img_col_all,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))
    # Add s2cloudless as new band
    img_col_all_join = img_col_all_join.map(lambda x: x.addBands(ee.Image(x.get('s2cloudless')).select('probability')))

    # Check no image is lost in the inner join
    n_images_join = img_col_all_join.size().getInfo()

    if n_images_join < n_images_col:
        if not force_s2cloudless:
            # Compute cloud mask from BQA band and add it as probability layer
            warnings.warn(
                f"Not all the images in the S2 collection {n_images_col} have s2cloudless cloud mask {n_images_join}. WE WILL IGNORE CLOUDS!")
            img_col_all_join = img_col_all.map(add_probability_from_BQA)

        else:
            warnings.warn(
                f"Not all the images in the S2 collection {n_images_col} have s2cloudless cloud mask {n_images_join}. We will skip images with missing cloud masks")

    daily_mosaic =  collection_mosaic_day(img_col_all_join, bounds)
                                    #fun_before_mosaic=lambda img: img.toFloat().resample("bicubic")) # Bicubic resampling for 60m res bands?

    # Add cloud info and valid info
    count_fun = get_count_function(bands, bounds)
    daily_mosaic = daily_mosaic.map(count_fun)

    return daily_mosaic
    # Filter images with many invalids

def get_count_function(bands, bounds):
    def _count_valid_clouds(img:ee.Image) -> ee.Image:
        mascara_valids = img.mask()
        mascara_valids = mascara_valids.select(bands)
        mascara_valids = mascara_valids.reduce(ee.Reducer.allNonZero())
        dictio = mascara_valids.reduceRegion(reducer=ee.Reducer.mean(), geometry=bounds,
                                             bestEffort=True, scale=10.)

        img = img.set("valids", dictio.get("all"))

        # Count clouds
        cloud_probability = img.select("probability")
        dictio = cloud_probability.reduceRegion(reducer=ee.Reducer.mean(), geometry=bounds,
                                                bestEffort=True, scale=10.)

        img = img.set("cloud_probability", dictio.get("probability"))

        return img

    return _count_valid_clouds


def collection_mosaic_day(imcol:ee.ImageCollection, region_of_interest:ee.Geometry)-> ee.ImageCollection:
    """
    Groups by solar day the images in the image collection. This function is useful to discard repeated images in
    image collections for example in the case of Sentinel-2 images.

    Args:
        imcol: image collection
        region_of_interest: needed to find the solar day

    Returns:

    """
    # https://gis.stackexchange.com/questions/280156/mosaicking-a-image-collection-by-date-day-in-google-earth-engine
    imlist = imcol.toList(imcol.size())

    # longitude, latitude = region_of_interest.centroid().coordinates().getInfo()
    longitude = region_of_interest.centroid().coordinates().get(0)

    hours_add = ee.Number(longitude).multiply(12/180.)
    # solar_time = utc_time - hours_add

    unique_solar_dates = imlist.map(lambda im: ee.Image(im).date().advance(hours_add, "hour").format("YYYY-MM-dd")).distinct()

    def mosaic_date(solar_date_str):
        solar_date = ee.Date(solar_date_str)
        utc_date = solar_date.advance(hours_add.multiply(-1), "hour")

        ims_day = imcol.filterDate(utc_date, utc_date.advance(1, "day"))

        dates = ims_day.toList(ims_day.size()).map(lambda x: ee.Image(x).date().millis())
        median_date = dates.reduce(ee.Reducer.median())

        im = ims_day.mosaic()

        im = im.copyProperties(ee.Image(ims_day.first()))

        return im.set({
            "system:time_start": median_date,
            "system:id": solar_date.format("YYYY-MM-dd"),
            "system:index": solar_date.format("YYYY-MM-dd")
        })

    mosaic_imlist = unique_solar_dates.map(mosaic_date)
    return ee.ImageCollection(mosaic_imlist)


PROPERTIES_DEFAULT = ["system:index", "system:time_start"]
def img_collection_to_feature_collection(img_col:ee.ImageCollection,
                                         properties:List[str]=PROPERTIES_DEFAULT,
                                         as_geopandas:bool=False) -> Union[ee.FeatureCollection, gpd.GeoDataFrame]:
    """Transforms the image collection to a feature collection """

    properties = ee.List(properties)

    def extractFeatures(img):
        values = properties.map(lambda prop: img.get(prop))
        dictio = ee.Dictionary.fromLists(properties, values)
        return ee.Feature(img.geometry(), dictio)

    feature_collection = ee.FeatureCollection(img_col.map(extractFeatures))
    if as_geopandas:
        geodf = gpd.GeoDataFrame.from_features(feature_collection.getInfo(), crs="EPSG:4326")
        if "system:time_start" in geodf.columns:
            geodf["datetime"] = pd.to_datetime(geodf["system:time_start"],unit="ms")
        return geodf

    return feature_collection


def istaskrunning(description:str) -> bool:
    task_list = ee.data.getTaskList()
    for t in task_list:
        if t["description"] == description:
            if (t["state"] == "READY") or (t["state"] == "RUNNING"):
                return True
    return False


def mayberun(filename, desc, function, export_task, overwrite=False, dry_run=False, verbose=1,
             bucket_name="worldfloods"):

    if bucket_name is not None:
        fs = get_filesystem("gs://")
    
        files_in_bucket = fs.glob(f'gs://{bucket_name}/{filename}*')
        if len(files_in_bucket) > 0:
            if overwrite:
                print("\tFile %s exists in the bucket. removing" % filename)
                for b in files_in_bucket:
                    fs.remove(f"gs://{b}")
            else:
                if verbose >= 2:
                    print(f"\tFile {filename} exists in the bucket, it will not be downloaded")
                return
    else:
        files = glob(f"{filename}*")
        if len(files) > 0:
            if overwrite:
                print("\tFile %s exists in the bucket. removing" % filename)
                for b in files:
                    os.remove(b)
            else:
                if verbose >= 2:
                    print(f"\tFile {filename} exists , it will not be downloaded")
                return

    if not dry_run and istaskrunning(desc):
        if verbose >= 2:
            print("\ttask %s already running!" % desc)
        return

    if dry_run:
        print("\tDRY RUN: Downloading file %s" % filename)
        return

    try:
        image_to_download = function()

        if image_to_download is None:
            return

        print("\tDownloading file %s" % filename)

        task = export_task(image_to_download, fileNamePrefix=filename, description=desc)

        task.start()

        return task

    except Exception:
        traceback.print_exc()

    return


def export_task_image(bucket:Optional[str]="worldfloods", crs:str='EPSG:4326',
                      scale:float=10, file_dims=16_384, maxPixels=5_000_000_000) -> Callable:
    """
    function to export images in the WorldFloods format.

    Args:
        bucket:
        scale:
        crs:
        file_dims:
        maxPixels:

    Returns:

    """

    if bucket is not None:
        def export_task(image_to_download, fileNamePrefix, description):
            task = ee.batch.Export.image.toCloudStorage(image_to_download,
                                                        fileNamePrefix=fileNamePrefix,
                                                        description=description,
                                                        crs=crs.upper(),
                                                        skipEmptyTiles=True,
                                                        bucket=bucket,
                                                        scale=scale,
                                                        fileFormat="GeoTIFF",
                                                        formatOptions={"cloudOptimized": True},
                                                        fileDimensions=file_dims,
                                                        maxPixels=maxPixels)
            return task
    else:
        def export_task(image_to_download, fileNamePrefix, description):
            task = ee.batch.Export.image.toDrive(image_to_download,
                                                 fileNamePrefix=fileNamePrefix,
                                                 description=description,
                                                 crs=crs.upper(),
                                                 skipEmptyTiles=True,
                                                 scale=scale,
                                                 fileFormat="GeoTIFF",
                                                 formatOptions={"cloudOptimized": True},
                                                 fileDimensions=file_dims,
                                                 maxPixels=maxPixels)
            return task

    return export_task


def generate_polygon(bbox:Tuple[float, float, float, float]) ->List[List[List[float]]]:
    """
    Generates a list of coordinates: [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x1,y1]]
    """
    return [[[bbox[0],bbox[1]],
             [bbox[2],bbox[1]],
             [bbox[2],bbox[3]],
             [bbox[0],bbox[3]],
             [bbox[0],bbox[1]]]]


def download_permanent_water(area_of_interest: Polygon, date_search:datetime,
                             path_bucket: str, crs:str='EPSG:4326',
                             name_task:Optional[str]=None, resolution_meters:int=10, requester_pays:bool = True) -> Optional[ee.batch.Task]:
    """
    Downloads yearly permanent water layer from the GEE. (JRC/GSW1_3/YearlyHistory product)

    Args:
        area_of_interest: polygon with the AoI to download
        date_search: start search date
        path_bucket: path in the bucket to export the image. If the files in that bucket exists it does not download
        them.
        crs: crs to export the images. To export them in utm based on location use the `convert_wgs_to_utm` function.
        name_task:
        resolution_meters:

    Returns:
        List of GEE tasks if triggered
    """
    assert path_bucket.startswith("gs://"), f"Path bucket: {path_bucket} must start with gs://"

    fs = get_filesystem("gs://")
    filename_full_path = os.path.join(path_bucket, f"{date_search.year}.tif")
    if fs.exists(filename_full_path):
        print(f"File {filename_full_path} exists. It will not be downloaded again")
        return

    ee.Initialize()
    
    path_bucket_no_gs = path_bucket.replace("gs://", "")
    bucket_name = path_bucket_no_gs.split("/")[0]
    path_no_bucket_name = "/".join(path_bucket_no_gs.split("/")[1:])

    area_of_interest_geojson = mapping(area_of_interest)
    pol = ee.Geometry(area_of_interest_geojson)
    bounding_box_aoi = area_of_interest.bounds
    bounding_box_pol = ee.Geometry.Polygon(generate_polygon(bounding_box_aoi))

    if date_search.year >= 2021:
        year = 2021
    else:
        year = date_search.year

    img_export = ee.Image(f"JRC/GSW1_4/YearlyHistory/{year}")

    if name_task is None:
        name_for_desc = os.path.basename(path_no_bucket_name)
    else:
        name_for_desc = name_task

    filename = os.path.join(path_no_bucket_name, f"{year}")
    desc = f"{name_for_desc}_{year}"

    export_task_fun_img = export_task_image(
        bucket=bucket_name,
        crs=crs,
        scale=resolution_meters,
    )

    return mayberun(
        filename,
        desc,
        lambda: img_export.clip(bounding_box_pol),
        export_task_fun_img,
        overwrite=False,
        dry_run=False,
        bucket_name=bucket_name,
        verbose=2,
    )


def permanent_water_image(year:int, pol:ee.Geometry) -> ee.Image:
    if year >= 2021:
        year = 2021
    
    img_export = ee.Image(f"JRC/GSW1_4/YearlyHistory/{year}")
    return img_export.clip(pol)


def download_merit_layer(area_of_interest: Polygon,
                         path_bucket: str, crs:str='EPSG:4326',
                         name_task:Optional[str]=None, resolution_meters:int=10) -> Optional[ee.batch.Task]:
    """
    Downloads MERIT Hydro product ("MERIT/Hydro/v1_0_1") from GEE

    Args:
        area_of_interest: polygon with the AoI to download
        path_bucket: path in the bucket to export the image. If the files in that bucket exists it does not download
        them.
        crs: crs to export the images. To export them in utm based on location use the `convert_wgs_to_utm` function.
        name_task:
        resolution_meters:

    Returns:
        List of GEE tasks if triggered
    """
    assert path_bucket.startswith("gs://"), f"Path bucket: {path_bucket} must start with gs://"

    fs = get_filesystem("gs://")
    
    filename_full_path = os.path.join(path_bucket, "merit").replace("\\",'/')
    if fs.exists(filename_full_path):
        print(f"File {filename_full_path} exists. It will not be downloaded again")
        return

    ee.Initialize()
    
    path_bucket_no_gs = path_bucket.replace("gs://", "")
    bucket_name = path_bucket_no_gs.split("/")[0]
    path_no_bucket_name = "/".join(path_bucket_no_gs.split("/")[1:])

    area_of_interest_geojson = mapping(area_of_interest)
    bounding_box_aoi = area_of_interest.bounds
    bounding_box_pol = ee.Geometry.Polygon(generate_polygon(bounding_box_aoi))

    img_export = ee.Image("MERIT/Hydro/v1_0_1").float()

    if name_task is None:
        name_for_desc = os.path.basename(path_no_bucket_name)
    else:
        name_for_desc = name_task

    filename = os.path.join(path_no_bucket_name, "merit").replace("\\",'/')
    
    desc = f"{name_for_desc}"

    export_task_fun_img = export_task_image(
        bucket=bucket_name,
        crs=crs,
        scale=resolution_meters,
    )
    

    return mayberun(
        filename,
        desc,
        lambda: img_export.clip(bounding_box_pol),
        export_task_fun_img,
        overwrite=False,
        dry_run=False,
        bucket_name=bucket_name,
        verbose=2,
    )



def convert_wgs_to_utm(lon: float, lat: float) -> str:
    """
    Based on lat and lng, return best utm epsg-code

    https://gis.stackexchange.com/questions/269518/auto-select-suitable-utm-zone-based-on-grid-intersection
    https://stackoverflow.com/questions/40132542/get-a-cartesian-projection-accurate-around-a-lat-lng-pair/40140326#40140326
    Args:
        lon:
        lat:

    Returns: string with the best utm espg-code

    """

    utm_band = str((math.floor((lon + 180) / 6 ) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0'+ utm_band
    if lat >= 0:
        epsg_code = 'EPSG:326' + utm_band
        return epsg_code
    epsg_code = 'EPSG:327' + utm_band
    return epsg_code


def process_metadata(path_csv:str, fs=None) -> pd.DataFrame:
    """
    Opens s2info.csv file that are exported in download_s2l89 function. It converts the date fields and
    adds a column indicating which files are available.

    Args:
        path_csv:
        fs:

    Returns:
        dataframe with processed date fields and column indicating if the s2 file is available
    """
    if fs is None:
        fs = get_filesystem("gs://")
    
    if path_csv.startswith("gs"):
        with fs.open(path_csv, "r") as fh:
            datas2 = pd.read_csv(fh)
    else:
        datas2 = pd.read_csv(path_csv)
        
                         # converters={'datetime': pd.Timestamp})

    datas2["datetime"] = datas2.datetime.apply(lambda x: datetime.fromisoformat(x).replace(tzinfo=timezone.utc))

    datas2["names2file"] = datas2.datetime.apply(lambda x: x.strftime("%Y-%m-%d"))
    datas2["s2available"] = datas2.names2file.apply(lambda x: fs.exists(os.path.join(os.path.dirname(path_csv),
                                                                                     x +".tif")))

    return datas2


def _check_all_downloaded(data:pd.DataFrame,
                          date_start_search: datetime, date_end_search: datetime,
                          filter_s2_fun:Optional[Callable[[pd.DataFrame], pd.Series]],
                          collection_name:str="S2") -> bool:
    """
    True if all images have been downloaded

    Args:
        data:
        date_start_search:
        date_end_search:
        filter_s2_fun:

    Returns:

    """

    min_date = min(data["datetime"])

    if (min_date > date_start_search) and ((min_date-date_start_search).total_seconds() / 3600.) > 10:
        return False

    max_date = max(data["datetime"])
    if (max_date < date_end_search) and ((date_end_search-max_date).total_seconds() / 3600.) > 10:
        return False

    if data.shape[0] <= 0:
        return False

    n_images_col = data.shape[0]
    if filter_s2_fun is not None:
        filter_good = filter_s2_fun(data)
        data = data[filter_good]
        images_available = data["s2available"].sum()
        print(
            f"Found {n_images_col} {collection_name} images between {date_start_search.isoformat()} and {date_end_search.isoformat()} "
            f"{np.sum(filter_good)} satisfies the criteria "
            f"{images_available} are already downloaded")
    else:
        images_available = data["s2available"].sum()
        print(f"Found {n_images_col} {collection_name} images between {date_start_search.isoformat()} and {date_end_search.isoformat()} "
              f"all satisfies the criteria "
              f"{images_available} are already downloaded")

    if data.shape[0] <= 0:
        return False

    if data["s2available"].all():
        return True

    return False


def download_s2l89(area_of_interest: Polygon,
                   date_start_search: datetime, date_end_search: datetime,
                   path_bucket: str,
                   collection_name="COPERNICUS/S2_HARMONIZED", crs:str='EPSG:4326',
                   filter_fun:Callable[[pd.DataFrame], pd.Series]=None,
                   name_task:Optional[str]=None,
                   force_s2cloudless:bool=True,
                   resolution_meters:float=10) -> List[ee.batch.Task]:
    """
    Download time series of S2 or Landsat images between search dates over the given area of interest. It saves the images
    on the path_bucket location. It only downloads images that satisfies the filter_s2 condition.

    Args:
        area_of_interest: shapely polygon with the AoI to download.
        date_start_search: start search date
        date_end_search: end search date
        path_bucket: path in the bucket to export the images. If the files in that bucket exists it does not download
        them.
        collection_name: "COPERNICUS/S2_HARMONIZED" for L1C Sentinel-2 images and ""COPERNICUS/S2_SR_HARMONIZED" for L2A images.
        crs: crs to export the images. To export them in utm based on location use the `convert_wgs_to_utm` function.
        filter_fun: function to filter the images to download. This function receives a dataframe with columns
            "cloud_probability", "valids" and "datetime" the output of this function should be boolean array of the
            with the number of rows of the dataframe that indicates which images of the dataframe to download.
        name_task: if not provided will use the basename of `path_bucket`
        force_s2cloudless:
        resolution_meters: resolution in meters to export the images

    Returns:
        List of running tasks and dataframe with metadata of the S2 files.

    """

    assert path_bucket.startswith("gs://"), f"Path bucket: {path_bucket} must start with gs://"

    path_bucket_no_gs = path_bucket.replace("gs://", "")
    bucket_name = path_bucket_no_gs.split("/")[0]
    path_no_bucket_name = "/".join(path_bucket_no_gs.split("/")[1:])

    fs = get_filesystem("gs://")

    if collection_name == "Landsat":
        path_csv = os.path.join(path_bucket, "landsatinfo.csv")
    elif collection_name == "S2":
        collection_name = "COPERNICUS/S2_HARMONIZED"
        path_csv = os.path.join(path_bucket, "s2info.csv")
    else:
        path_csv = os.path.join(path_bucket, "s2info.csv")

    if fs.exists(path_csv):
        data = process_metadata(path_csv, fs=fs)
        if _check_all_downloaded(data, date_start_search=date_start_search,
                                 date_end_search=date_end_search,
                                 filter_s2_fun=filter_fun,
                                 collection_name=collection_name):
            return []
        else:
            min_date = min(data["datetime"])
            max_date = max(data["datetime"])
            date_start_search = min(min_date, date_start_search)
            date_end_search = max(max_date, date_end_search)

    ee.Initialize()
    area_of_interest_geojson = mapping(area_of_interest)
    bounding_box_aoi = area_of_interest.bounds
    bounding_box_pol = ee.Geometry.Polygon(generate_polygon(bounding_box_aoi))

    pol = ee.Geometry(area_of_interest_geojson)

    # Grab the S2 images
    if collection_name == "Landsat":
        img_col = get_landsat_collection(date_start_search, date_end_search, pol, verbose=2)
    else:
        img_col = get_s2_collection(date_start_search, date_end_search, pol,
                                    bands=BANDS_NAMES[collection_name],
                                    force_s2cloudless=force_s2cloudless,
                                    collection_name=collection_name, verbose=2)
    if img_col is None:
        return []

    # Get info of the S2 images (convert to table)
    img_col_info_local = image_collection_fetch_metadata(img_col)

    n_images_col = img_col_info_local.shape[0]

    # Save S2 images as csv
    with fs.open(path_csv, "wb") as fh:
        img_col_info_local.to_csv(fh, index=False, mode="wb")

    print(f"Found {n_images_col} {collection_name} images between {date_start_search.isoformat()} and {date_end_search.isoformat()}")

    imgs_list = img_col.toList(n_images_col, 0)

    export_task_fun_img = export_task_image(
        bucket=bucket_name,
        crs=crs,
        scale=resolution_meters,
    )
    if filter_fun is not None:
        filter_good = filter_fun(img_col_info_local)

        if np.sum(filter_good) == 0:
            print("All images are bad")
            return []

        img_col_info_local_good = img_col_info_local[filter_good]
    else:
        img_col_info_local_good = img_col_info_local

    tasks = []
    for good_images in img_col_info_local_good.itertuples():
        img_export = ee.Image(imgs_list.get(good_images.index_image_collection))
        img_export = img_export.select(BANDS_NAMES[collection_name] + ["probability"]).toUint16().clip(bounding_box_pol)

        date = good_images.datetime.strftime('%Y-%m-%d')

        if name_task is None:
            name_for_desc = os.path.basename(path_no_bucket_name)
        else:
            name_for_desc = name_task
        
        filename = os.path.join(path_no_bucket_name, date)
        desc = f"{name_for_desc}_{date}"
        task = mayberun(
            filename,
            desc,
            lambda: img_export,
            export_task_fun_img,
            overwrite=False,
            dry_run=False,
            bucket_name=bucket_name,
            verbose=2,
        )
        if task is not None:
            tasks.append(task)

    return tasks


def image_collection_fetch_metadata(img_col:ee.ImageCollection) -> pd.DataFrame:
    """
    Return the metadata of the provided image collection as a pandas dataframe.
    """
    img_col_info = img_collection_to_feature_collection(img_col,
                                                        ["system:time_start", "valids",
                                                         "cloud_probability"])
    img_col_info_local = gpd.GeoDataFrame.from_features(img_col_info.getInfo())
    img_col_info_local["datetime"] = img_col_info_local["system:time_start"].apply(
        lambda x: datetime.utcfromtimestamp(x / 1000).replace(tzinfo=timezone.utc))
    img_col_info_local["cloud_probability"] /= 100
    img_col_info_local = img_col_info_local[["system:time_start", "valids", "cloud_probability", "datetime"]]
    img_col_info_local["index_image_collection"] = np.arange(img_col_info_local.shape[0])
    return img_col_info_local


def wait_tasks(tasks:List[ee.batch.Task]) -> None:
    """
    Wait for a list of tasks to finish

    Args:
        tasks: list of ee.Tasks

    """
    task_down = []
    for task in tasks:
        if task.active():
            task_down.append((task.status()["description"],task))

    task_error = 0
    while len(task_down) > 0:
        print("%d tasks running" % len(task_down))

        task_down_new = []
        for _i, (t, task) in enumerate(list(task_down)):
            if task.active():
                task_down_new.append((t, task))
                continue
            if task.status()["state"] != "COMPLETED":
                print("Error in task {}:\n {}".format(t, task.status()))
                task_error += 1

        task_down = task_down_new
        time.sleep(60)

    print("Tasks failed: %d" % task_error)
