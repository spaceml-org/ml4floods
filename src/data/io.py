import numpy as np
from typing import Optional, Dict
import rasterio


def save_groundtruth_tiff_rasterio(
    image_gt: np.ndarray,
    destination_path: str,
    gt_meta: Optional[Dict] = None,
    **kwargs
) -> None:
    """Save image as tiff with rasterio

    Args:
        image (np.ndarray): image to be saved
            image should be of size (n_channels, height, width)
        destination_path (str): path where the image is saved
        gt_meta Optional[Dict]: a dictionary of extra metadata to be saved in
            the tags
        **kwargs

    Examples:
        >>> destination_path = "./temp.tif"
        >>> # path to image for coordinates
        >>> with rasterio.open(s2_image_path) as src_s2:
        >>>    crs = src_s2.crs
        >>>    transform = src_s2.transform
        >>> save_groundtruth_tiff_rasterio(
            gt, destination_path,
            transform=transform, crs=crs
        )
    """

    # get image channels
    if image_gt.ndim != 3:
        image_gt = image_gt[None, ...]

    n_channels, height, width = image_gt.shape

    # write the image
    with rasterio.open(
        destination_path,
        "w",
        driver="COG",
        height=height,
        width=width,
        count=n_channels,
        dtype=image_gt.dtype,
        **kwargs
    ) as dst:

        if gt_meta is not None:
            dst.update_tags(gt_meta=gt_meta)
        dst.write(image_gt)
    return None