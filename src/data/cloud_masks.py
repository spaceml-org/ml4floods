from s2cloudless import S2PixelCloudDetector
import rasterio
import argparse
import os
import numpy as np


def sentinel2_to_cloud_mask_preprocess(x):
    """
    takes x in the format of the tif file and rescales it to the format that s2 cloudless expects.
    """
    # last channel is a 'quality assesment' channel rather than a sensor input
    # s2 cloudless also expects channels last and to be scaled to 0-1
    
    return x[:13, :, :].transpose(1, 2, 0)[None, ...] / 10000

def compute_cloud_mask(x):
    z = sentinel2_to_cloud_mask_preprocess(x)
    cloud_detector = S2PixelCloudDetector(
        threshold=0.4, average_over=4, dilation_size=2, all_bands=True
    )

    cloud_mask = cloud_detector.get_cloud_probability_maps(z)
    cloud_mask = cloud_mask.squeeze()
    return cloud_mask

def compute_cloud_mask_save(cp_path, x, profile):
    cloud_mask = compute_cloud_mask(x)

    if cp_path is not None:
        profile.update(count=1, compress="lzw", dtype="float32", driver="COG",
                       BIGTIFF= "IF_SAFER",RESAMPLING="CUBICSPLINE") # Generate overviews with CUBICSPLINE resampling!

        with rasterio.open(cp_path, "w", **profile) as dst:
            dst.write(cloud_mask.astype(np.float32), 1)

    return cloud_mask


def create_cloud_mask(tf_path, cp_path, verbose):
    with rasterio.open(tf_path) as x_tif:
        x = x_tif.read()
        compute_cloud_mask_save(cp_path, x, x_tif.profile, verbose)

    return


def main(worldfloods_root, verbose):
    copernicus_s2_path = os.path.join(worldfloods_root, "tiffimages", "S2")

    copernicus_cloudprob_path = os.path.join(
        worldfloods_root, "tiffimages", "cloudprob"
    )
    if not os.path.exists(copernicus_cloudprob_path):
        os.mkdir(copernicus_cloudprob_path)

    tiffiles = os.listdir(copernicus_s2_path)

    for i, tf in enumerate(tiffiles):
        tf_path = os.path.join(copernicus_s2_path, tf)
        cp_path = os.path.join(copernicus_cloudprob_path, tf)
        if i % 20 == 0:
            print('')
        if verbose:
            print("PROCESSING:", tf_path, cp_path)
        if os.path.exists(cp_path):
#             print(cp_path, "already exists, skipping...")
            print(',', end='', flush=True)
            continue
        try:
            print('.', end='',flush=True)
            create_cloud_mask(tf_path, cp_path, verbose)
        except Exception as e:
            print(f'problem in file {tf_path}: {e}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("worldfloods_root", help="directory where bucket is mounted")
    parser.add_argument(
        "-v", "--verbose", help="turn on verbose mode", action="store_true"
    )

    args = parser.parse_args()

    main(args.worldfloods_root, args.verbose)
