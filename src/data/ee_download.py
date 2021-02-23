import ee
from google.cloud import storage
import traceback
import requests
import shutil
import tempfile

import geemap.eefolium

def get_collection(collection_name, date_start, date_end, bounds):
    collection = ee.ImageCollection(collection_name)
    collection_filtered = collection.filterDate(date_start, date_end) \
        .filterBounds(bounds)

    n_images = int(collection_filtered.size().getInfo())

    return collection_filtered, n_images


def collection_mosaic_day(imcol, region_of_interest):
    """
    Groups by solar day the images in the image collection.

    :param imcol:
    :param region_of_interest:
    :return:
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

        im = ims_day.mosaic()
        # im = im.reproject(ee.Projection("EPSG:3857"), scale=10)
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
        print("\tDownloading file %s" % filename)
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