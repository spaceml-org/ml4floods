import shutil
import tempfile
import traceback

import ee
import time
import requests
from google.cloud import storage
import os
from glob import glob
from typing import Optional, Callable, List, Tuple
from shapely.geometry import mapping, Polygon
import numpy as np
import geopandas as gpd
import pandas as pd
import fsspec
from ml4floods.data.config import BANDS_S2
from datetime import datetime, timezone


BANDS_S2_NAMES = {
    # Sentinel-2 L1C
    "COPERNICUS/S2" : ["B1","B2","B3","B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12", "QA60"],
    # Sentinel-2 L2A
    "COPERNICUS/S2_SR" : ["B1","B2","B3","B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "SCL"]
}


def permanent_water_image(year, bounds):
    # permananet water files are only available pre-2019
    if year >= 2020:
        year = 2020
    return ee.Image(f"JRC/GSW1_3/YearlyHistory/{year}").clip(bounds)


def get_collection(collection_name, date_start, date_end, bounds):
    collection = ee.ImageCollection(collection_name)
    collection_filtered = collection.filterDate(date_start, date_end) \
        .filterBounds(bounds)

    n_images = int(collection_filtered.size().getInfo())

    return collection_filtered, n_images


def get_s2_collection(date_start, date_end, bounds, collection_name="COPERNICUS/S2", bands=BANDS_S2, verbose=1,
                      threshold_invalid=.5):
    """

    Args:
        date_start:
        date_end:
        bounds:
        collection_name:
        bands:
        threshold_invalid:

    Returns:

    """
    img_col_all, n_images_col = get_collection(collection_name, date_start, date_end, bounds)
    if n_images_col <= 0:
        if verbose > 1:
            print(f"Not images found for collection {collection_name} date start: {date_start} date end: {date_end}")
        return

    img_col_all = img_col_all.select(bands)

    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(bounds)
        .filterDate(date_start, date_end))

    img_col_all = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': img_col_all,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))

    # Add s2cloudless as new band
    img_col_all = img_col_all.map(lambda x: x.addBands(ee.Image(x.get('s2cloudless')).select('probability')))

    daily_mosaic =  collection_mosaic_day(img_col_all, bounds,
                                          fun_before_mosaic=None)
                                    #fun_before_mosaic=lambda img: img.toFloat().resample("bicubic")) # Bicubic resampling for 60m res bands?

    # Filter images with many invalids
    def _count_valid_clouds(img):
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

    daily_mosaic = daily_mosaic.map(_count_valid_clouds).filter(ee.Filter.greaterThanOrEquals('valids', threshold_invalid))

    return daily_mosaic


def collection_mosaic_day(imcol, region_of_interest, fun_before_mosaic=None):
    """
    Groups by solar day the images in the image collection.

    Args:
        imcol:
        region_of_interest:
        fun_before_mosaic:

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

        # im = ims_day.mosaic()
        if fun_before_mosaic is not None:
            ims_day = ims_day.map(fun_before_mosaic)

        im = ims_day.mosaic()
        return im.set({
            "system:time_start": median_date,
            "system:id": solar_date.format("YYYY-MM-dd"),
            "system:index": solar_date.format("YYYY-MM-dd")
        })

    mosaic_imlist = unique_solar_dates.map(mosaic_date)
    return ee.ImageCollection(mosaic_imlist)


PROPERTIES_DEFAULT = ["system:index", "system:time_start"]
def img_collection_to_feature_collection(img_col, properties=PROPERTIES_DEFAULT):
    properties = ee.List(properties)

    def extractFeatures(img):
        values = properties.map(lambda prop: img.get(prop))
        dictio = ee.Dictionary.fromLists(properties, values)
        return ee.Feature(img.geometry(), dictio)

    return ee.FeatureCollection(img_col.map(extractFeatures))


def export_eeFeatureCollection(feature_col, properties=None, filename=None, filetype="GeoJSON"):
    """
    exports ee.FeatureCollection to a file to be read.

    :param feature_col: feature collection to export
    :type feature_col: ee.FeatureCollection
    :param properties: (optional) list of columns to export
    :param filename: (optional) name of the file.
    :param filetype: type of the file.

    :return: filename to open (with geopandas if geojson)
    """
    if filename is None:
        fileobj = tempfile.NamedTemporaryFile(dir=".", suffix=f".{filetype.lower()}", delete=True)
        filename = fileobj.name
        fileobj.close()

    if properties is None:
        url = feature_col.getDownloadURL(filetype=filetype)
    else:
        properties_list = properties.getInfo()
        url = feature_col.getDownloadURL(filetype=filetype,
                                         selectors=properties_list)

    r_link = requests.get(url, stream=True)
    if r_link.status_code == 200:
        with open(filename, 'wb') as f:
            r_link.raw.decode_content = True
            shutil.copyfileobj(r_link.raw, f)

    return filename


def findtask(description):
    task_list = ee.data.getTaskList()
    for t in task_list:
        if t["description"] == description:
            if (t["state"] == "READY") or (t["state"] == "RUNNING"):
                return True
    return False


def mayberun(filename, desc, function, export_task, overwrite=False, dry_run=False, verbose=1,
             bucket_name="worldfloods"):

    if bucket_name is not None:
        bucket = storage.Client().get_bucket(bucket_name)
        blobs_rasterized_geom = list(bucket.list_blobs(prefix=filename))

        if len(blobs_rasterized_geom) > 0:
            if overwrite:
                print("\tFile %s exists in the bucket. removing" % filename)
                for b in blobs_rasterized_geom:
                    b.delete()
            else:
                if verbose >= 2:
                    print("\tFile %s exists in the bucket, it will not be downloaded" % filename)
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

    if not dry_run and findtask(desc):
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


def export_task_image(bucket=Optional["worldfloods"],crs='EPSG:4326',
                      scale=10, file_dims=16_384, maxPixels=5_000_000_000) -> Callable:
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
                                                 formatOptions={"cloudOptimized": True},
                                                 fileDimensions=file_dims,
                                                 maxPixels=maxPixels)
            return task

    return export_task


def export_task_featurecollection(bucket="worldfloods", fileFormat="GeoJSON"):
    def export_task(featcol2down, fileNamePrefix, description):
        task = ee.batch.Export.table.toCloudStorage(featcol2down,
                                                    fileNamePrefix=fileNamePrefix,
                                                    description=description,
                                                    fileFormat=fileFormat,
                                                    bucket=bucket)
        return task

    return export_task


def bbox_2_eepolygon(bbox):
    # ee.Polygon must be in long,lat
    return ee.Geometry.Polygon([[[bbox["west"], bbox["north"]],
                                 [bbox["east"], bbox["north"]],
                                 [bbox["east"], bbox["south"]],
                                 [bbox["west"], bbox["south"]]]])


def download_permanent_water(area_of_interest: Polygon, date_search:datetime,
                             path_bucket: str, crs:str='EPSG:4326',
                             name_task:Optional[str]=None, resolution_meters:int=10) -> Optional[ee.batch.Task]:
    """
    Downloads yearly permanent water layer from the GEE. (JRC/GSW1_3/YearlyHistory product)
    Args:
        area_of_interest:
        date_search:
        path_bucket:
        crs:
        name_task:
        resolution_meters:

    Returns:

    """
    assert path_bucket.startswith("gs://"), f"Path bucket: {path_bucket} must start with gs://"

    fs = fsspec.filesystem("gs")
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

    img_export = permanent_water_image(date_search.year, pol)

    if name_task is None:
        name_for_desc = os.path.basename(path_no_bucket_name)
    else:
        name_for_desc = name_task

    filename = os.path.join(path_no_bucket_name, f"{date_search.year}")
    desc = f"{name_for_desc}_{date_search.year}"

    export_task_fun_img = export_task_image(
        bucket=bucket_name,
        crs=crs,
        scale=resolution_meters,
    )

    return mayberun(
        filename,
        desc,
        lambda: img_export,
        export_task_fun_img,
        overwrite=False,
        dry_run=False,
        bucket_name=bucket_name,
        verbose=2,
    )


def process_s2metadata(path_csv:str, fs=None) -> pd.DataFrame:
    if fs is None:
        fs = fsspec.filesystem("gs")

    datas2 = pd.read_csv(path_csv)
                         # converters={'datetime': pd.Timestamp})

    datas2["datetime"] = datas2.datetime.apply(lambda x: datetime.fromisoformat(x).replace(tzinfo=timezone.utc))

    datas2["names2file"] = datas2.datetime.apply(lambda x: x.strftime("%Y-%m-%d"))
    datas2["s2available"] = datas2.names2file.apply(lambda x: fs.exists(os.path.join(os.path.basename(path_csv),
                                                                                     x +".tif")))

    return datas2


def check_rerun(data:pd.DataFrame,
                date_start_search: datetime, date_end_search: datetime,
                filter_s2_fun:Optional[Callable[[pd.DataFrame], pd.Series]]) -> bool:
    """ Check if any S2 image is missing to trigger download """

    min_date = min(data["datetime"])

    if (min_date > date_start_search) and ((min_date-date_start_search).total_seconds() / 3600.) > 10:
        return False

    max_date = max(data["datetime"])
    if (max_date < date_end_search) and ((date_end_search-max_date).total_seconds() / 3600.) > 10:
        return False

    if data.shape[0] <= 0:
        return False

    if filter_s2_fun is not None:
        filter_good = filter_s2_fun(data)
        data = data[filter_good]

    if data.shape[0] <= 0:
        return False

    if data["s2available"].all():
        return True

    return False


def download_s2(area_of_interest: Polygon,
                date_start_search: datetime, date_end_search: datetime,
                path_bucket: str, collection_name="COPERNICUS/S2_SR", crs:str='EPSG:4326',
                filter_s2_fun:Callable[[pd.DataFrame], pd.Series]=None,
                name_task=None,
                resolution_meters=10) -> List[ee.batch.Task]:
    """
    Download time series of S2 images between search dates over the given area of interest. It saves the S2 images on
    path_bucket location. It only downloads images with less than threshold_invalid invalid pixels and with less than
    threshold_clouds cloudy pixels.

    Args:
        area_of_interest:
        date_start_search:
        date_end_search:
        path_bucket:
        collection_name:
        crs:
        filter_s2_fun:
        name_task:
        resolution_meters:

    Returns:
        List of running tasks and dataframe with metadata of the S2 files.

    """

    assert path_bucket.startswith("gs://"), f"Path bucket: {path_bucket} must start with gs://"

    path_bucket_no_gs = path_bucket.replace("gs://", "")
    bucket_name = path_bucket_no_gs.split("/")[0]
    path_no_bucket_name = "/".join(path_bucket_no_gs.split("/")[1:])

    fs = fsspec.filesystem("gs")
    path_csv = os.path.join(path_bucket, "s2info.csv")
    if fs.exists(path_csv):
        data = process_s2metadata(path_csv, fs=fs)
        if not check_rerun(data, date_start_search=date_start_search,
                           date_end_search=date_end_search,
                           filter_s2_fun=filter_s2_fun):
            return []
        else:
            min_date = min(data["datetime"])
            max_date = max(data["datetime"])
            date_start_search = min(min_date, date_start_search)
            date_end_search = max(max_date, date_end_search)

    ee.Initialize()
    area_of_interest_geojson = mapping(area_of_interest)

    pol = ee.Geometry(area_of_interest_geojson)

    # Grab the S2 images
    img_col = get_s2_collection(date_start_search, date_end_search, pol,
                                collection_name=collection_name)
    if img_col is None:
        return []

    # Get info of the S2 images (convert to table)
    img_col_info = img_collection_to_feature_collection(img_col,
                                                        ["system:time_start", "valids",
                                                         "cloud_probability"])

    img_col_info_local = gpd.GeoDataFrame.from_features(img_col_info.getInfo())
    img_col_info_local["datetime"] = img_col_info_local["system:time_start"].apply(
        lambda x: datetime.utcfromtimestamp(x / 1000))
    img_col_info_local["cloud_probability"] /= 100
    img_col_info_local = img_col_info_local[["system:time_start", "valids", "cloud_probability", "datetime"]]
    img_col_info_local["index_image_collection"] = np.arange(img_col_info_local.shape[0])

    n_images_col = img_col_info_local.shape[0]

    # Save S2 images as csv
    img_col_info_local.to_csv(path_csv)

    print(f"Found {n_images_col} S2 images between {date_start_search.isoformat()} and {date_end_search.isoformat()}")

    imgs_list = img_col.toList(n_images_col, 0)

    export_task_fun_img = export_task_image(
        bucket=bucket_name,
        crs=crs,
        scale=resolution_meters,
    )
    if filter_s2_fun is not None:
        filter_good = filter_s2_fun(img_col_info_local)

        if np.sum(filter_good) == 0:
            print("All images are bad")
            return []

        img_col_info_local_good = img_col_info_local[filter_good]
    else:
        img_col_info_local_good = img_col_info_local

    tasks = []
    for good_images in img_col_info_local_good.itertuples():
        img_export = ee.Image(imgs_list.get(good_images.index_image_collection))
        img_export = img_export.select(BANDS_S2_NAMES[collection_name] + ["probability"]).toFloat().clip(pol)

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


def wait_tasks(tasks:List[ee.batch.Task]) -> None:
    task_down = []
    for task in tasks:
        if task.active():
            task_down.append((task.status()["description"],task))

    task_error = 0
    while len(task_down) > 0:
        print("%d tasks running" % len(task_down))
        for _i, (t, task) in enumerate(list(task_down)):
            if task.active():
                continue
            if task.status()["state"] != "COMPLETED":
                print("Error in task {}:\n {}".format(t, task.status()))
                task_error += 1
            del task_down[_i]

        time.sleep(60)

    print("Tasks failed: %d" % task_error)