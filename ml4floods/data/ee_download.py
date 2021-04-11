import shutil
import tempfile
import traceback

import ee
import requests
from google.cloud import storage

from ml4floods.data.config import BANDS_S2


def download_permanent_water(date, bounds):
    
    year = date.year    
    # permananet water files are only available pre-2019
    if year >= 2019:
        year = 2019
    return ee.Image(f"JRC/GSW1_2/YearlyHistory/{year}").clip(bounds)


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
        mascara = img.mask()
        mascara = mascara.select(bands)
        mascara = mascara.reduce(ee.Reducer.allNonZero())
        dictio = mascara.reduceRegion(reducer=ee.Reducer.mean(), geometry=bounds,
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

        # im = ims_day.mosaic()
        if fun_before_mosaic is not None:
            ims_day = ims_day.map(fun_before_mosaic)

        im = ims_day.mosaic()
        return im.set({
            "system:time_start": utc_date.millis(),
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


def export_task_image(bucket="worldfloods", scale=10, file_dims=12544, maxPixels=5000000000):
    def export_task(image_to_download, fileNamePrefix, description):
        task = ee.batch.Export.image.toCloudStorage(image_to_download,
                                                    fileNamePrefix=fileNamePrefix,
                                                    description=description,
                                                    crs='EPSG:4326',
                                                    skipEmptyTiles=True,
                                                    bucket=bucket,
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