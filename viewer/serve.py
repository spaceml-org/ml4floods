from flask import Flask, send_file
import os
import argparse
import json
import io
from typing import List
from PIL import Image
import numpy as np
import rasterio
import rasterio.windows
from rasterio import warp
from ml4floods.data.config import BANDS_S2

from glob import glob
import mercantile


from contextlib import ExitStack

STATIC_FOLDER = "web"
app = Flask(__name__, static_url_path='/web')
app.config['STATIC_FOLDER'] = STATIC_FOLDER

SATURATION = 3_000
SIZE = 256
WEB_MERCATOR_CRS = "epsg:3857"

PIXEL_PRECISION = 3
# Required because we cant read in parallel with rasterio
def pad_window(window: rasterio.windows.Window, pad_size) -> rasterio.windows.Window:
    """ Add the provided pad to a rasterio window object """
    return rasterio.windows.Window(window.col_off - pad_size[1],
                                   window.row_off - pad_size[0],
                                   width=window.width + 2 * pad_size[1],
                                   height=window.height + 2 * pad_size[1])


def round_outer_window(window:rasterio.windows.Window)-> rasterio.windows.Window:
    """ Rounds a rasterio.windows.Window object to outer (larger) window """
    return window.round_lengths(op="ceil", pixel_precision=PIXEL_PRECISION).round_offsets(op="floor",
                                                                                          pixel_precision=PIXEL_PRECISION)


@app.route("/<subset>/<eventid>/<productname>/<z>/<x>/<y>.png")
def servexyz(subset:str, eventid:str, productname:str, z, x, y):
    """
    A route to get an RGB JPEG clipped from a given geotiff Sentinel-2 image for a given z,x,y TMS tile coordinate.
    Parameters
    """

    ### get latlon bounding box from z,x,y tile request
    bounds_wgs = mercantile.xy_bounds(int(x), int(y), int(z))

    image_address = os.path.join(app.config["ROOT_LOCATION"], subset, productname, f"{eventid}.tif")

    bands = [1, 2] if productname == "gt" else [BANDS_S2.index(b) + 1 for b in ["B11", "B8", "B4"]]
    resampling = warp.Resampling.nearest if productname == "gt" else warp.Resampling.cubic_spline

    OUTPUT_SHAPE = (len(bands), SIZE, SIZE)

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN=True, CPL_DEBUG=True), rasterio.open(image_address, 'r') as rst:
        x1, y1, x2, y2 = rst.bounds
        rst_bounds = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
        if rst.crs != WEB_MERCATOR_CRS:
            bounds_read_crs_dest = warp.transform_bounds({"init": WEB_MERCATOR_CRS}, rst.crs,
                                                         left=bounds_wgs.left,
                                                         top=bounds_wgs.top,
                                                         bottom=bounds_wgs.bottom, right=bounds_wgs.right)
        else:
            bounds_read_crs_dest = (bounds_wgs.left, bounds_wgs.bottom, bounds_wgs.right, bounds_wgs.top)

        if rasterio.coords.disjoint_bounds(bounds_read_crs_dest, rst_bounds):
            return app.send_static_file('border.png')

        rst_transform = rst.transform
        rst_crs = rst.crs
        rst_nodata = rst.nodata or 0
        rst_dtype = rst.meta["dtype"]

        # Compute the window to read and read the data
        window = rasterio.windows.from_bounds(*bounds_read_crs_dest, rst.transform)
        if rst.crs != WEB_MERCATOR_CRS:
            window = pad_window(window, (1, 1))  # Add padding for bicubic int or for co-registration
            window = round_outer_window(window)

        OUTPUT_SHAPE_READ = (len(bands), SIZE+1, SIZE+1)

        src_arr = rst.read(bands, window=window, out_shape=OUTPUT_SHAPE_READ, boundless=True,
                           fill_value=0)

    # Compute transform of readed data and reproject to WEB_MERCATOR_CRS
    input_output_factor = (window.height / OUTPUT_SHAPE_READ[1], window.width / OUTPUT_SHAPE_READ[2])

    transform_window = rasterio.windows.transform(window, rst_transform)
    transform_src = rasterio.Affine(transform_window.a * input_output_factor[1], transform_window.b, transform_window.c,
                                    transform_window.d, transform_window.e * input_output_factor[0], transform_window.f)

    transform_dst = rasterio.transform.from_bounds(west=bounds_wgs.left,north=bounds_wgs.top,
                                                   east=bounds_wgs.right, south=bounds_wgs.bottom,
                                                   width=SIZE, height=SIZE)

    rst_arr = np.empty(shape=OUTPUT_SHAPE, dtype=rst_dtype)

    rst_arr, rst_transform = warp.reproject(source=src_arr, destination=rst_arr,
                                            src_transform=transform_src, src_crs=rst_crs,
                                            src_nodata=rst_nodata,
                                            dst_transform=transform_dst, dst_crs=WEB_MERCATOR_CRS,
                                            dst_nodata=0,
                                            resampling=resampling)

    alpha = (~np.all(rst_arr == 0, axis=0)).astype(np.uint8) * 255

    if productname == "S2":
        img_rgb = (np.clip(rst_arr / SATURATION, 0, 1).transpose((1, 2, 0)) * 255).astype(np.uint8)
    else:
        clear_clouds = rst_arr[0]
        land_water = rst_arr[1]

        v1gt = land_water.copy()  # {0: invalid, 1: land, 2: water}
        v1gt[clear_clouds == 2] = 3
        img_rgb = mask_to_rgb(v1gt, [0, 1, 2, 3], colores=COLORS)

    img_rgb = np.concatenate([img_rgb, alpha[...,None]], axis=-1)

    buf = io.BytesIO()
    Image.fromarray(img_rgb, mode="RGBA").save(buf, format="PNG")
    buf.seek(0, 0)

    return send_file(
        buf,
        as_attachment=True,
        download_name=f'{z}_{x}_{y}.png',
        mimetype='image/png'
    )


COLORS = np.array([[0, 0, 0], # invalid
                   [139, 64, 0], # land
                   [0, 0, 139], # water
                   [220, 220, 220]], # cloud
                  dtype=np.uint8)


@app.route('/worldfloods.json')
def worldfloods_database():
    return send_file(app.config["DATABASE_NAME"])


def mask_to_rgb(mask: np.ndarray, values: List[int], colores: np.ndarray) -> np.ndarray:
    """
    Given a 2D mask it assign each value of the mask the corresponding color

    :param mask: (H, W)
    :param values: (K,)
    :param colores: (K, d) d could be 4 if mode is RGBA or 3
    :return: (H, W, d)
    """
    assert len(values) == len(colores), "Values and colors should have same length {} {}".format(len(values),len(colores))

    mask_return = np.zeros(mask.shape[:2] + (colores.shape[-1],), dtype=np.uint8)
    colores = np.array(colores)
    for i, c in enumerate(colores):
        mask_return[mask == values[i], :] = c

    return mask_return


@app.route('/')
def root():
    return app.send_static_file('index.html')

# KEYS_COPY_V2 = ["satellite", "event id", "satellite date", "ems_code", "aoi_code", "date_ems_code", "s2_date"]
KEYS_COPY = ["satellite", "satellite date", "s2_date"]

def worldfloods_files(rl:str):

    json_files = sorted(glob(os.path.join(rl, "*/meta/*.json")))
    worldfloods = []

    for json_file in json_files:
        with open(json_file, "r") as fh:
            meta =  json.load(fh)

        # TODO handle date_s2 stuff between formats
        meta_copy = {k:meta[k] for k in KEYS_COPY}
        meta_copy["subset"] = os.path.basename(os.path.dirname(os.path.dirname(json_file)))
        meta_copy["geometry"] = meta["area_of_interest_polygon"]

        worldfloods.append({"id": os.path.splitext(os.path.basename(json_file))[0], "meta": meta_copy})

    return worldfloods


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch web page to inspect data')
    parser.add_argument('--port', type=int,
                        default=3142,
                        help='port to run')
    # parser.add_argument('--debug', help="Debug",
    #                     action="store_true")
    parser.add_argument('--host',  type=str, required=False, help="Use \"0.0.0.0\" to have "
                                                                  "the server available externally as well")
    parser.add_argument('--root_location', help='Root folder', type=str,
                        default='/media/disk/databases/WORLDFLOODS/2_Mart/worldfloods_extra_v2_0/')

    args = parser.parse_args()

    root_location = args.root_location[:-1] if args.root_location.endswith("/") else args.root_location
    database_name = os.path.basename(root_location)

    pdb = os.path.join("web", database_name+".json")

    if not os.path.exists(pdb):
        print(f"Generate database from location {args.root_location}")
        database = worldfloods_files(root_location)

        database[1]["selected"] = True
        with open(pdb, "w") as fh:
            json.dump(database, fh)
    else:
        print(f"Loading database: {pdb}")


    app.static_folder = app.config["STATIC_FOLDER"]
    app.config["ROOT_LOCATION"] = os.path.abspath(args.root_location)
    app.config["DATABASE_NAME"] = os.path.abspath(pdb)

    app.run(port=args.port, debug=True, host=args.host, threaded=False)


