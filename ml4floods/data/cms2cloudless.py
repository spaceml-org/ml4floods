import argparse
import os

from typing import Optional
import numpy as np
import rasterio
from s2cloudless import S2PixelCloudDetector
from ml4floods.data.worldfloods.configs import BANDS_S2


def sentinel2_to_cloud_mask_preprocess(x):
    """
    takes x in the format of the tif file and rescales it to the format that s2 cloudless expects.
    """
    # last channel is a 'quality assesment' channel rather than a sensor input
    # s2 cloudless also expects channels last and to be scaled to 0-1

    return x[:13, :, :].transpose(1, 2, 0)[None, ...] / 10000


def compute_cloud_mask(x: np.ndarray, threshold: float=0.4, average_over: int=4, dilation_size: int=2, all_bands: bool=True):
    z = sentinel2_to_cloud_mask_preprocess(x)
    cloud_detector = S2PixelCloudDetector(
        threshold=threshold, 
        average_over=average_over, 
        dilation_size=dilation_size, 
        all_bands=all_bands
    )

    cloud_mask = cloud_detector.get_cloud_probability_maps(z)
    cloud_mask = cloud_mask.squeeze()
    return cloud_mask


def compute_cloud_mask_save(cp_path, x, profile):
    cloud_mask = compute_cloud_mask(x)

    if cp_path is not None:
        profile.update(
            count=1,
            compress="lzw",
            dtype="float32",
            driver="COG",
            BIGTIFF="IF_SAFER",
            RESAMPLING="CUBICSPLINE",
        )  # Generate overviews with CUBICSPLINE resampling!

        with rasterio.open(cp_path, "w", **profile) as dst:
            dst.write(cloud_mask.astype(np.float32), 1)

    return cloud_mask


def compute_s2cloudless_probs(s2_image_path: str, window: Optional[rasterio.windows.Window]=None, **kwargs) -> np.ndarray:
    bands_read = list(range(1, len(BANDS_S2) + 1))
    
    # open the S2 Image
    with rasterio.open(s2_image_path, "r") as s2_rst:
        s2_img = s2_rst.read(bands_read, window=window)
        
    return compute_cloud_mask(s2_img, **kwargs)

def create_cloud_mask(tf_path, cp_path, verbose):
    with rasterio.open(tf_path) as x_tif:
        x = x_tif.read()
        compute_cloud_mask_save(cp_path, x, x_tif.profile, verbose)

    return


def main(worldfloods_root, verbose):
    copernicus_s2_path = os.path.join(worldfloods_root, "tiffimages", "S2").replace("\\","/")

    copernicus_cloudprob_path = os.path.join(
        worldfloods_root, "tiffimages", "cloudprob"
    ).replace("\\","/")
    if not os.path.exists(copernicus_cloudprob_path):
        os.mkdir(copernicus_cloudprob_path)

    tiffiles = os.listdir(copernicus_s2_path)

    for i, tf in enumerate(tiffiles):
        tf_path = os.path.join(copernicus_s2_path, tf).replace("\\","/")
        cp_path = os.path.join(copernicus_cloudprob_path, tf).replace("\\","/")
        if i % 20 == 0:
            print("")
        if verbose:
            print("PROCESSING:", tf_path, cp_path)
        if os.path.exists(cp_path):
            #             print(cp_path, "already exists, skipping...")
            print(",", end="", flush=True)
            continue
        try:
            print(".", end="", flush=True)
            create_cloud_mask(tf_path, cp_path, verbose)
        except Exception as e:
            print(f"problem in file {tf_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("worldfloods_root", help="directory where bucket is mounted")
    parser.add_argument(
        "-v", "--verbose", help="turn on verbose mode", action="store_true"
    )

    args = parser.parse_args()

    main(args.worldfloods_root, args.verbose)
