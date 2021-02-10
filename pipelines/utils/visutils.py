import json
import numpy as np
import rasterio
from PIL import Image
import os

def exists_bucket_or_disk(path):
    """ Function that checks if a path file exists to be used as drop-in replace for os.path.exists """
    if path.startswith("gs://"):
        from google.cloud import storage
        splitted_path = path.replace("gs://","").split("/")
        bucket_name = splitted_path[0]
        blob_name = "/".join(splitted_path[1:])
        buck = storage.bucket(bucket_name)
        return buck.get_blob(blob_name) is not None
    return os.path.exists(path)


def generate_rgb(s2tiff, name_full_path=None):
    """ generate rgb from S2 image if it does not exists """

    if name_full_path is None:
        if s2tiff.startswith("gs://"):
            raise NotImplementedError("Does not save directly in the bucket")
        name_full_path = s2tiff.replace("/tiffimages/","/rgbimages/").replace(".tif", ".jpg")
        name_full_path = name_full_path.replace("/S2/", "/S2rgb/")

    if os.path.exists(name_full_path):
        # File exists do not recompute
        return

    with rasterio.open(s2tiff) as s2rst:
        s2img = s2rst.read((4, 3, 2))
    s2img_rgb = (np.clip(s2img / 2500, 0, 1).transpose((1, 2, 0)) * 255).astype(np.uint8)

    Image.fromarray(s2img_rgb).save(name_full_path, format="JPEG")


def generate_mask_rgb(gt, name_full_path):
    """ Given gt raster coded as {0: invalid, 1: land, 2: water, 3: cloud} it generates a jpeg file for viz """

    gt_rgb = mask_to_rgb(gt, values=[0, 1, 2, 3],
                         colors_cmap=np.array([[0, 0, 0],
                                               [139, 64, 0],
                                               [0, 0, 139],
                                               [220, 220, 220]]) / 255)
    Image.fromarray(gt_rgb).save(name_full_path, format="JPEG")


def mask_to_rgb(mask, values, colors_cmap):
    """
    Given a 2D mask it assign each value of the mask the corresponding color
    :param mask:
    :param values:
    :param colors_cmap:
    :return:
    """
    assert len(values) == len(colors_cmap), "Values and colors should have same length {} {}".format(len(values), len(colors_cmap))

    mask_return = np.zeros(mask.shape[:2] + (3,), dtype=np.uint8)
    colores = np.array(np.round(colors_cmap*255), dtype=np.uint8)
    for i, c in enumerate(colores):
        mask_return[mask == values[i], :] = c

    return mask_return


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj_to_encode):
        """Pandas and Numpy have some specific types that we want to ensure
        are coerced to Python types, for JSON generation purposes. This attempts
        to do so where applicable.
        """
        # Pandas dataframes have a to_json() method, so we'll check for that and
        # return it if so.
        if hasattr(obj_to_encode, 'to_json'):
            return obj_to_encode.to_json()
        # Numpy objects report themselves oddly in error logs, but this generic
        # type mostly captures what we're after.
        if isinstance(obj_to_encode, np.generic):
            return obj_to_encode.item()
        # ndarray -> list, pretty straightforward.
        if isinstance(obj_to_encode, np.ndarray):
            return obj_to_encode.tolist()
        # torch or tensorflow -> list, pretty straightforward.
        if hasattr(obj_to_encode, "numpy"):
            return obj_to_encode.numpy().tolist()
        # If none of the above apply, we'll default back to the standard JSON encoding
        # routines and let it work normally.
        return super().default(obj_to_encode)
