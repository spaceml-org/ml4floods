from rasterio import features
import rasterio
from shapely.geometry import shape, mapping, Polygon
from skimage import measure
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import numpy as np
from typing import List, Optional

from ml4floods.data.create_gt import get_brightness, BRIGHTNESS_THRESHOLD

def preprocess_water_probabilities(prob_water_mask, thres_h=0.6, thres_l=0.4, distance= 5, conn=2, watershed_line=True):
    mask0 = prob_water_mask > thres_h
    local_maxi = peak_local_max(prob_water_mask, indices=False, footprint=np.ones((distance * 2 + 1, distance * 2 + 1)),
                                labels=(prob_water_mask > thres_l))
    local_maxi[mask0] = True
    seed_msk = ndi.label(local_maxi)[0]

    prob_water_mask = watershed(-prob_water_mask, seed_msk, mask=(prob_water_mask > thres_l), watershed_line=watershed_line)
    return measure.label(prob_water_mask, connectivity=conn, background=0).astype('uint8')


def get_water_polygons(binary_water_mask: np.ndarray, min_area:float=25.5,
                       polygon_buffer:int=0, tolerance:float=1., transform: Optional[rasterio.Affine]=None) -> List[Polygon]:
    """

    Args:
        binary_water_mask: (H, W) binary mask to rasterise
        min_area: polygons with pixel area lower than this will be filtered
        polygon_buffer: buffering of the polygons
        tolerance: to simplify the polygons
        transform: affine transformation of the binary_water_mask raster

    Returns:
        list of rasterised polygons

    """
    assert binary_water_mask.ndim == 2, f"Expected mask with 2 dim found {binary_water_mask.shape}"

    geoms_polygons = []
    polygon_generator = features.shapes(binary_water_mask.astype(np.int16),
                                        binary_water_mask)

    for polygon, value in polygon_generator:
        p = shape(polygon)
        if polygon_buffer > 0:
            p = p.buffer(polygon_buffer)
        if p.area >= min_area:
            p = p.simplify(tolerance=tolerance)
            if transform is not None:
                p = transform_polygon(p, transform) # Convert polygon to raster coordinates
            geoms_polygons.append(p)

    return geoms_polygons

def transform_polygon(polygon:Polygon, transform: rasterio.Affine) -> Polygon:
    """
    Transforms a polygon from pixel coordinates to the coordinates specified by the affine transform

    Args:
        polygon: polygon to transform
        transform: Affine transformation

    Returns:
        polygon with coordinates transformed by the affine transformation

    """
    geojson_dict = mapping(polygon)
    out_coords = []
    for pol in geojson_dict["coordinates"]:
        pol_out = []
        for coords in pol:
            pol_out.append(transform * coords)

        out_coords.append(pol_out)

    geojson_dict["coordinates"] = out_coords

    return shape(geojson_dict)

def get_pred_mask_v2(inputs: np.ndarray, prediction: np.ndarray, 
                     th_water:int = 0.5, th_cloud:int = 0.5, mask_clouds:bool = True) -> np.ndarray:
    """
    Receives an output of a WFV2 model (multioutput binary) and returns the corresponding segmentation mask 

    Args:
        inputs: S2 image (B, H, W)
        prediction: corresponding model output(2, H, W)
        th_water: threshold for the class water
        th_cloud: threshhold for the class cloud
        mask_clouds: If False ignores britghtness and outputs the prediction mask with thin clouds
        
    Returns:
        Water mask (H, W) with interpretation {0: invalids, 1: land, 2: water, 3: cloud}

    """
    mask_invalids = np.all(inputs==0, axis = 0).squeeze()
    cloud_mask = (prediction[0] > th_cloud).astype(np.float64)
    water_mask = (prediction[1] > th_water).astype(np.float64)
    water_mask += 1
    
    if mask_clouds:
        br = get_brightness(inputs)
        br_th = br > BRIGHTNESS_THRESHOLD
        water_mask[(cloud_mask==1) & (br_th == 1)] = 3
    else:
        water_mask[cloud_mask==1] = 3
    
    water_mask[mask_invalids] = 0
    return water_mask