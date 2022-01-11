from flask import Flask, send_file
import os
import argparse
import json
import io
from typing import List
from PIL import Image
import numpy as np
import geopandas
from ml4floods.data.config import BANDS_S2
from ml4floods.serve.read_tile import read_tile
from rasterio import warp

from glob import glob

STATIC_FOLDER = "web"
app = Flask(__name__, static_url_path='/web')
app.config['STATIC_FOLDER'] = STATIC_FOLDER

SATURATION = 3_000


@app.route("/<subset>/<eventid>/floodmap.geojson")
def floodmap(subset:str, eventid:str):
    floodmap_address = os.path.join(app.config["ROOT_LOCATION"], subset, "floodmaps", f"{eventid}.geojson")
    data = geopandas.read_file(floodmap_address).to_crs("epsg:4326")
    data = data[data["source"] != "area_of_interest"]
    data["id"] = np.arange(data.shape[0])

    buf = io.BytesIO()
    data.to_file(buf, driver="GeoJSON")
    buf.seek(0,0)
    return send_file(buf,
                     as_attachment=True,
                     download_name=f'{subset}_{eventid}_floodmap.geojson',
                     mimetype='application/geojson'
                     )


@app.route("/<subset>/<eventid>/<productname>/<z>/<x>/<y>.png")
def servexyz(subset:str, eventid:str, productname:str, z, x, y):
    """
    A route to get an RGB JPEG clipped from a given geotiff Sentinel-2 image for a given z,x,y TMS tile coordinate.
    """

    image_address = os.path.join(app.config["ROOT_LOCATION"], subset, productname, f"{eventid}.tif")

    bands = [1, 2] if productname == "gt" else [BANDS_S2.index(b) + 1 for b in ["B11", "B8", "B4"]]
    resampling = warp.Resampling.nearest if productname == "gt" else warp.Resampling.cubic_spline

    output = read_tile(image_address, x=int(x),  y=int(y), z=int(z), indexes=bands,
                       resampling=resampling, dst_nodata=0)

    if output is None:
        # Not intersects return None
        return app.send_static_file("border.png")

    rst_arr, rst_transform = output

    alpha = (~np.all(rst_arr == 0, axis=0)).astype(np.uint8) * 255

    if productname == "S2":
        img_rgb = (np.clip(rst_arr / SATURATION, 0, 1).transpose((1, 2, 0)) * 255).astype(np.uint8)
    else:
        clear_clouds = rst_arr[0]
        land_water = rst_arr[1]

        v1gt = land_water.copy()  # {0: invalid, 1: land, 2: water}
        v1gt[clear_clouds == 2] = 3
        img_rgb = mask_to_rgb(v1gt, [0, 1, 2, 3], colors=COLORS)

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


def mask_to_rgb(mask: np.ndarray, values: List[int], colors: np.ndarray) -> np.ndarray:
    """
    Given a 2D mask it assigns each value of the mask in `values` the corresponding color in `colors`

    Args:
        mask: (H, W)
        values: (K,)
        colors: (K, d) d could be 4 if mode is RGBA or 3

    Returns:
        (H, W, d)

    """
    assert len(values) == len(colors), "Values and colors should have same length {} {}".format(len(values), len(colors))

    mask_return = np.zeros(mask.shape[:2] + (colors.shape[-1],), dtype=np.uint8)
    colors = np.array(colors)
    for i, c in enumerate(colors):
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


