import ee
from google.cloud import storage
import traceback


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