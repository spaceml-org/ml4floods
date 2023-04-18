import pandas as pd
from flask import Flask, send_file, request
import os
import argparse
import json
import io
from typing import List, Optional
from PIL import Image
import numpy as np
import geopandas
from ml4floods.data.worldfloods.configs import BANDS_S2
from ml4floods.serve.read_tile import read_tile
from rasterio import warp, features
from ml4floods.data import utils, create_gt, save_cog, vectorize
from shapely.geometry import shape
import geopandas as gpd
import logging

from glob import glob

STATIC_FOLDER = "web"
app = Flask(__name__, static_url_path='/web')
app.config['STATIC_FOLDER'] = STATIC_FOLDER

DRIVER_FLOODMAPS = {
    "geojson": "GeoJSON",
    "shp" : "ESRI Shapefile"
}
SATURATION = 3_500

CLOUDFOLDER = "cloud_vec"

def generate_gt_v2(clouds: np.ndarray,
                   water_mask: np.ndarray,
                   invalids: np.ndarray) -> np.ndarray:
    cloudgt = np.ones(water_mask.shape, dtype=np.uint8)  # whole image is 1 -> clear
    cloudgt[clouds >= 1] = 2  # only cloud is 2
    cloudgt[invalids] = 0

    watergt = np.ones(water_mask.shape, dtype=np.uint8)  # whole image is 1 -> land
    watergt[water_mask >= 1] = 2  # only water is 2
    watergt[invalids] = 0
    watergt[cloudgt == 2] = 0

    stacked_cloud_water_mask = np.stack([cloudgt, watergt], axis=0)

    return stacked_cloud_water_mask

def generate_gt_v1(clouds: np.ndarray,
                   water_mask: np.ndarray,
                   invalids: np.ndarray) -> np.array:
    water_mask = water_mask[None]
    gt = np.ones(water_mask.shape, dtype=np.uint8)
    gt[water_mask >= 1] = 2
    gt[clouds] = 3
    gt[invalids] = 0
    return


@app.route("/<subset>/<eventid>/save_floodmap", methods = ['POST'])
def save_floodmap(subset:str, eventid:str):
    meta_path = os.path.join(app.config["ROOT_LOCATION"], subset, "meta", f"{eventid}.json")
    meta = utils.read_json_from_gcp(meta_path)

    if "area_of_interest_polygon" in meta:
        aoi = shape(meta["area_of_interest_polygon"])
    else:
        # Assumes old version of worldfloods
        aoi = box(*meta["bounds"])


    geojson = request.json
    floodmap = gpd.GeoDataFrame.from_features(geojson, crs="epsg:4326")

    # remove column id
    floodmap = floodmap[["geometry", "w_class", "source"]]

    # Intersect all geometries with aoi
    floodmap["geometry"] = floodmap.geometry.apply(lambda x: x.intersection(aoi))

    # Drop NA/empty geometries
    floodmap = floodmap[(~floodmap.geometry.isna()) & (~floodmap.geometry.is_empty)]

    # add AoI
    floodmap = floodmap.append({"geometry": aoi, "w_class": "area_of_interest", "source": "area_of_interest"},
                               ignore_index=True)
    floodmap = floodmap.set_crs(epsg=4326)

    # reproject to S2 crs
    s2path = os.path.join(app.config["ROOT_LOCATION"], subset, "S2", f"{eventid}.tif")
    with utils.rasterio_open_read(s2path) as src:
        crs = str(src.crs)

    floodmap.to_crs(crs, inplace=True)

    # separate cloudmap
    cloudmap = floodmap[floodmap["w_class"] == "cloud"].copy()
    floodmap = floodmap[floodmap["w_class"] != "cloud"].copy()

    # Save floodmap locally
    floodmap_path = os.path.join(app.config["ROOT_LOCATION"], subset, "floodmaps", f"{eventid}.{app.config['FORMAT_FLOODMAPS']}")
    os.makedirs(os.path.dirname(floodmap_path), exist_ok=True)
    floodmap.to_file(floodmap_path, driver=DRIVER_FLOODMAPS[app.config['FORMAT_FLOODMAPS']])

    # Save cloudmap locally
    cloudmap = cloudmap.rename({"w_class": "class"}, axis=1)
    cloudmap = cloudmap[["geometry","class"]]
    cloudmap_path = os.path.join(app.config["ROOT_LOCATION"], subset, CLOUDFOLDER,
                                 f"{eventid}.geojson")
    os.makedirs(os.path.dirname(cloudmap_path), exist_ok=True)
    utils.write_geojson_to_gcp(cloudmap_path, cloudmap)

    # Recompute gt
    jrc_permanent_water_path = os.path.join(app.config["ROOT_LOCATION"], subset, "PERMANENTWATERJRC", f"{eventid}.tif")
    gt_version = app.config["GT_VERSION"]
    water_mask = create_gt.compute_water(s2path,floodmap=floodmap,permanent_water_path=jrc_permanent_water_path,
                                         keep_streams=gt_version == "v2")
    current_gt_path = os.path.join(app.config["ROOT_LOCATION"], subset, "gt", f"{eventid}.tif")
    with utils.rasterio_open_read(current_gt_path) as rst:
        current_gt = rst.read()
        transform = rst.transform
        crs = rst.crs
        tags = rst.tags()

    invalids = (current_gt[0] == 0) | (water_mask == -1)

    # rasterise cloudmap
    if cloudmap.shape[0] > 0:
        shapes_rasterise = (
            (g, 1)
            for g in cloudmap["geometry"] if g and not g.is_empty
        )
        clouds = features.rasterize(
            shapes=shapes_rasterise,
            fill=0,
            out_shape=water_mask.shape[-2:],
            dtype=np.uint8,
            transform=transform,
            all_touched=True
        )
    else:
        clouds = np.zeros(water_mask.shape[-2:], dtype=np.uint8)

    if gt_version == "v2":
        current_gt = generate_gt_v2(clouds, water_mask, invalids)
    else:
        current_gt = generate_gt_v1(clouds, water_mask, invalids)

    # TODO update tags and meta with the number of valid pixels etc??

    save_cog.save_cog(current_gt, current_gt_path,
                      {"crs": crs, "transform": transform, "RESAMPLING": "NEAREST",
                       "compress": "lzw", "nodata": 0},  # In both gts 0 is nodata
                      tags=tags)

    if app.config["SAVE_FLOODMAP_BUCKET"]:
        # save floodmap in stagging
        stagging_path = f"gs://ml4cc_data_lake/0_DEV/1_Staging/WorldFloods/{meta['ems_code']}/{meta['aoi_code']}/floodmap_edited/{meta['satellite date'][:10]}.geojson"
        utils.write_geojson_to_gcp(stagging_path, floodmap)
        logging.info(f"Saving file in {stagging_path}")

        #  save cloudmap in stagging
        stagging_clouds_path = f"gs://ml4cc_data_lake/0_DEV/1_Staging/WorldFloods/{meta['ems_code']}/{meta['aoi_code']}/cmedited_vec/{meta['names2file']}.geojson"
        utils.write_geojson_to_gcp(stagging_clouds_path, cloudmap)
        logging.info(f"Saving file in {stagging_clouds_path}")

    return '', 204


@app.route("/<subset>/<eventid>/<predname>.geojson")
def read_floodmap_pred(subset:str, eventid:str, predname:str):
    # WF2_unet_full_norm_vec
    floodmap_address = os.path.join(app.config["ROOT_LOCATION"], subset, predname,"S2",
                                    f"{eventid}.{app.config['FORMAT_FLOODMAPS']}")

    data = geopandas.read_file(floodmap_address)

    # All parts of a simplified geometry will be no more than tolerance distance from the original
    if data.crs != "epsg:4326":
        data["geometry"] = data["geometry"].simplify(tolerance=10)

    # Set columns to ground truth columns
    data = data.rename({"class": "w_class"},
                       axis=1)
    data["source"] = predname
    data = data[data["w_class"] != "area_imaged"].copy()

    data.to_crs("epsg:4326", inplace=True)
    # data["id"] = np.arange(data.shape[0])

    buf = io.BytesIO()
    data.to_file(buf, driver="GeoJSON")

    buf.seek(0,0)
    return send_file(buf,
                     as_attachment=True,
                     download_name=f'{subset}_{eventid}_{predname}.geojson',
                     mimetype='application/geojson')


@app.route("/<subset>/<eventid>/floodmap.geojson")
def read_floodmap(subset:str, eventid:str):
    floodmap_path = os.path.join(app.config["ROOT_LOCATION"], subset, "floodmaps",
                                 f"{eventid}.{app.config['FORMAT_FLOODMAPS']}")
    data = geopandas.read_file(floodmap_path)

    cloudmap_path = os.path.join(app.config["ROOT_LOCATION"],
                                  subset, CLOUDFOLDER, f"{eventid}.geojson")

    if not os.path.exists(cloudmap_path):
        # load GT
        current_gt_path = os.path.join(app.config["ROOT_LOCATION"], subset, "gt", f"{eventid}.tif")
        with utils.rasterio_open_read(current_gt_path) as rst:
            current_gt = rst.read()
            transform = rst.transform
            crs = rst.crs

        if app.config["GT_VERSION"] == "v2":
            clouds = current_gt[0] == 2
        else:
            clouds = current_gt[0] == 3

        if np.any(clouds):
            geoms_polygons = vectorize.get_polygons(clouds,
                                                    transform=transform)
        else:
            geoms_polygons = []

        if len(geoms_polygons) > 0:
            cloudmap = gpd.GeoDataFrame({"geometry": geoms_polygons,
                                         "class": "cloud"},
                                        crs=crs)
        else:
            cloudmap = gpd.GeoDataFrame(data={"class": []},
                                          geometry=[], crs=crs)
        # save vectorized cloudprob
        os.makedirs(os.path.dirname(cloudmap_path), exist_ok=True)
        utils.write_geojson_to_gcp(cloudmap_path, cloudmap)

    else:
        cloudmap = geopandas.read_file(cloudmap_path)

    # Concat cloudmap to floodmap
    if cloudmap.shape[0] > 0:
        cloudmap.to_crs(data.crs, inplace=True)
        cloudmap["source"] = "cloud_vec"
        cloudmap = cloudmap.rename({"class": "w_class"},
                                   axis=1)

        cloudmap = cloudmap[["geometry","w_class", "source"]]
        data = pd.concat([data, cloudmap], ignore_index=True)

    data = data[data["source"] != "area_of_interest"]
    data = data[(~data.geometry.isna()) & (~data.geometry.is_empty)]

    # flatten MultiPolygons remove empty pols
    data = data.explode(ignore_index=True)

    # All parts of a simplified geometry will be no more than tolerance distance from the original
    if data.crs != "epsg:4326":
        data["geometry"] = data["geometry"].simplify(tolerance=10)
        data.to_crs("epsg:4326", inplace=True)

    data["id"] = np.arange(data.shape[0])

    buf = io.BytesIO()
    data.to_file(buf, driver="GeoJSON")
    buf.seek(0,0)
    return send_file(buf,
                     as_attachment=True,
                     download_name=f'{subset}_{eventid}_floodmap.geojson',
                     mimetype='application/geojson')


@app.route("/<subset>/<eventid>/<productname>/<z>/<x>/<y>.png")
def servexyz(subset:str, eventid:str, productname:str, z, x, y):
    """
     A route to get an RGB PNG from a given geotiff image for a given z,x,y TMS tile coordinate.

    Args:
        subset: {"train", "test", "val"}
        eventid: name of the event (e.g EMSR342_07SOUTHNORMANTON_DEL_MONIT03_v2)
        productname: {"S2RGB", "S2SWIRNIRRED", "gt", "PERMANENTWATERJRC", "WF2_unet_full_norm",
                      "MNDWI", "BRIGHTNESS", "gtcloud"}
        z: zoom level (max zoom for Sentinel-2 is 14)
        x: x location mercantile
        y: y location mercantile

    Returns:
        PNG of shape 256x256

    """

    productnamefolder = productname
    if productname.startswith("S2"):
        band_composite = productname.replace("S2","")
        if band_composite == "RGB":
            bands = [BANDS_S2.index(b) + 1 for b in ["B4", "B3", "B2"]]
        else:
            bands =[BANDS_S2.index(b) + 1 for b in ["B11", "B8", "B4"]]
        productname = "S2"
        productnamefolder = "S2"
        resampling = warp.Resampling.cubic_spline
    elif productname == 'S1':
        bands = [1]
        resampling = warp.Resampling.cubic_spline
    elif productname == "gt":
        if app.config["GT_VERSION"] == "v2":
            bands = [2]
        else:
            bands = [1]
        resampling = warp.Resampling.nearest
    elif productname == "gtcloud":
        productnamefolder = "gt"
        bands = [1]
        resampling = warp.Resampling.nearest
    elif productname == "MNDWI":
        bands = [BANDS_S2.index(b) + 1 for b in ["B11", "B3"]]
        resampling = warp.Resampling.cubic_spline
        productnamefolder = "S2"
    elif productname == "BRIGHTNESS":
        bands = [BANDS_S2.index(b) + 1 for b in ["B4", "B3", "B2"]]
        resampling = warp.Resampling.cubic_spline
        productnamefolder = "S2"
    elif productname == "PERMANENTWATERJRC":
        bands = [1]
        resampling = warp.Resampling.nearest
    elif productname == "WF2_unet_full_norm":
        productnamefolder = "WF2_unet_full_norm/S2"
        bands = [1]
        resampling = warp.Resampling.nearest
    else:
        raise NotImplementedError(f"Productname {productname} not found")


    image_address = os.path.join(app.config["ROOT_LOCATION"], subset, productnamefolder, f"{eventid}.tif")

    if not os.path.exists(image_address):
        logging.error(f"{image_address} does not exist")
        return '', 204

    output = read_tile(image_address, x=int(x),  y=int(y), z=int(z), indexes=bands,
                       resampling=resampling, dst_nodata=0)

    if output is None:
        # Not intersects return None
        return '', 204

    rst_arr, _ = output
    # rst_arr = np.nan_to_num(rst_arr, copy=False, nan=0)

    if productname == "S2":
        alpha = (~np.all(rst_arr == 0, axis=0)).astype(np.uint8) * 255
        img_rgb = (np.clip(rst_arr / SATURATION, 0, 1).transpose((1, 2, 0)) * 255).astype(np.uint8)
        img_rgb = np.concatenate([img_rgb, alpha[..., None]], axis=-1)
        mode = "RGBA"
    if productname == "S1":
        img_rgb = return_scaled_s1(rst_arr)
        mode = "RGB"
    elif productname in ["gt","WF2_unet_full_norm"]:
        pred = rst_arr[0]
        img_rgb = mask_to_rgb(pred, [0, 1, 2, 3], colors=COLORS)
        mode = "RGB"
    elif productname == "gtcloud":
        pred = rst_arr[0]
        if app.config["GT_VERSION"] == "v1":
            img_rgb = mask_to_rgb(pred, [0, 1, 2, 3], colors=COLORS)
        else:
            img_rgb = mask_to_rgb(pred, [0, 1, 2], colors=COLORS[[0,1,3]])

        mode = "RGB"
    elif productname == "MNDWI":
        invalid = np.all(rst_arr == 0, axis=0)
        rst_arr = rst_arr.astype(np.float32)
        band_sum = rst_arr[1] + rst_arr[0]
        band_diff = rst_arr[1] - rst_arr[0]
        dwi = band_diff / (band_sum + 1e-6)
        dwi_threshold = (dwi > 0).astype(np.uint8) + 1
        dwi_threshold[invalid] = 0
        img_rgb = mask_to_rgb(dwi_threshold, [0, 1, 2], colors=COLORS[:-1])
        mode = "RGB"
    elif productname == "BRIGHTNESS":
        invalid = np.all(rst_arr == 0, axis=0)
        brightness = create_gt.get_brightness(rst_arr, [1, 2, 3])
        brightness_threshold = (brightness >= create_gt.BRIGHTNESS_THRESHOLD).astype(np.uint8) + 1
        brightness_threshold[invalid] = 0
        img_rgb = mask_to_rgb(brightness_threshold, [0, 1, 2], colors=COLORS[(0,1, 3),...])
        mode = "RGB"
    elif productname == "PERMANENTWATERJRC":
        permanent_water = rst_arr[0]
        permanent_water = (permanent_water == 3).astype(np.uint8)

        img_rgb = mask_to_rgb(permanent_water, [0, 1], colors=COLORS[(0, 2), ...])

        # Make transparent all pixels except permanent water
        valids = permanent_water * 255
        img_rgb = np.concatenate([img_rgb, valids[..., None]], axis=-1)
        mode = "RGBA"
    else:
        raise NotImplementedError(f"Productname {productname} not found")


    buf = io.BytesIO()
    Image.fromarray(img_rgb, mode=mode).save(buf, format="PNG")
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

def return_scaled_s1(s1, return_scaling = False):
    
    ratio = s1[0] / s1[1] + 1e-4
    rgb_s1 = np.concatenate([s1[0][np.newaxis], s1[1][np.newaxis],ratio[np.newaxis]], axis = 0)

    mins = np.array([np.nanmin(rgb_s1[0]), np.nanmin(rgb_s1[0]), np.nanmin(ratio)])
    maxs = np.array([np.nanmax(rgb_s1[0]), np.nanmax(rgb_s1[0]), np.nanmax(ratio)])
    
    rgb_s1 = ( rgb_s1 - mins[0][None,None]) / (maxs[0][None,None] - mins[0][None,None])
    rgb_s1 = np.clip(rgb_s1,0,1)
    
    if return_scaling:
        return np.moveaxis(rgb_s1, 0,-1), mins, maxs
    else:
        return np.moveaxis(rgb_s1, 0,-1)


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


from shapely.geometry import box, mapping

# KEYS_COPY_V2 = ["satellite", "event id", "satellite date", "ems_code", "aoi_code", "date_ems_code", "s2_date"]
KEYS_COPY = ["satellite", "satellite date"]
def worldfloods_files(rl:str, status:Optional[pd.DataFrame]=None):

    json_files = sorted(glob(os.path.join(rl, "*/meta/*.json")))
    worldfloods = []
    if status is not None:
        status = status.set_index('layer name')

    for json_file in json_files:
        json_file = json_file.replace("\\", "/")
        layer_name = os.path.splitext(os.path.basename(json_file))[0]

        with open(json_file, "r") as fh:
            meta =  json.load(fh)

        meta_copy = {k: meta[k] for k in KEYS_COPY}
        if "s2_date" in meta:
            meta_copy["s2_date"] = meta["s2_date"]
        else:
            # Assumes old version of worldfloods
            meta_copy["s2_date"] = meta["s2metadata"][0]["date_string"]

        meta_copy["date_ems_code"] = meta.get("date_ems_code", "UNKNOWN")
        meta_copy["subset"] = os.path.basename(os.path.dirname(os.path.dirname(json_file)))
        if status is not None:
            if layer_name in status.index:
                meta_copy["subset_name"] = status.loc[layer_name, "subset"]
                status_iter = status.loc[layer_name, "status"]
                if (meta_copy["subset_name"] in ["unused", "train"]) and status_iter == 1:
                    meta_copy["subset_name"] = "train"
                elif (meta_copy["subset_name"] in ["unused", "train"]) and status_iter == 2:
                    meta_copy["subset_name"] = "to-fix"
                elif (meta_copy["subset_name"] in ["unused", "train"]) and status_iter == 0:
                    meta_copy["subset_name"] = "discarded"
            else:
                print(f"Layer {layer_name} not found in status, we will set it to unused")
                meta_copy["subset_name"] = "unused"
        else:
            meta_copy["subset_name"] = meta_copy["subset"]

        if "area_of_interest_polygon" in meta:
            meta_copy["geometry"] = meta["area_of_interest_polygon"]
        else:
            # Assumes old version of worldfloods
            meta_copy["geometry"] = mapping(box(*meta["bounds"]))

        worldfloods.append({"id": layer_name, "meta": meta_copy})

    return worldfloods


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch web page to inspect data')
    parser.add_argument('--port', type=int,
                        default=3142,
                        help='port to run')
    parser.add_argument('--host',  type=str, required=False, help="Use \"0.0.0.0\" to have "
                                                                  "the server available externally as well")
    parser.add_argument('--root_location', help='Root folder', type=str,
                        default='/media/disk/databases/WORLDFLOODS/2_Mart/worldfloods_extra_v2_0_DEF/')
    parser.add_argument("--gt_version", default="v2", choices=["v1", "v2"],
                        help="Version of ground truth. v1 1 band 3 classes, v2 2 bands 2 classes each")
    parser.add_argument('--no_save_floodmap_bucket', help="Do not save the floodmaps to the bucket", action="store_true")
    # add argument for status.csv file
    parser.add_argument('--status_csv', help='status csv file', type=str,
                        default='/media/disk/databases/WORLDFLOODS/Database_DEF.csv')
    

    args = parser.parse_args()
    
    if not args.no_save_floodmap_bucket:
        assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"], "GOOGLE_APPLICATION_CREDENTIALS env varible not set. " \
                                                             "This is needed to save floodmaps! in gs://ml4cc_data_lake\n" \
                                                             "If you don't want to save them use --no_save_floodmap_bucket option"

    root_location = args.root_location[:-1] if args.root_location.endswith("/") else args.root_location
    database_name = os.path.basename(root_location)

    pdb = os.path.join("web", f"{database_name}.json")

    # read status csv
    if os.path.exists(args.status_csv):
        status = pd.read_csv(args.status_csv, index_col=0)
    else:
        status = None

    print(f"Generate database from location {args.root_location}")
    database = worldfloods_files(root_location, status)

    database[1]["selected"] = True
    with open(pdb, "w") as fh:
        json.dump(database, fh)

    geojson_files = glob(os.path.join(root_location, "*/floodmaps/*.geojson"))
    if len(geojson_files) > 0:
        format_floodmaps = "geojson"
    else:
        format_floodmaps = "shp"

    app.static_folder = app.config["STATIC_FOLDER"]
    app.config["ROOT_LOCATION"] = os.path.abspath(args.root_location)
    app.config["DATABASE_NAME"] = os.path.abspath(pdb)
    app.config["GT_VERSION"] = args.gt_version
    app.config["FORMAT_FLOODMAPS"] = format_floodmaps
    app.config["SAVE_FLOODMAP_BUCKET"] = not args.no_save_floodmap_bucket

    # gunicorn core.asgi:application -w ${NUMBER_OF_WORKERS:-1} -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000

    app.run(port=args.port, debug=True, host=args.host, threaded=False)

