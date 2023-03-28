import numpy as np
from typing import Optional, List
import rasterio
from shapely.geometry import Polygon, shape, mapping
from rasterio import features


def get_polygons(binary_water_mask: np.ndarray, min_area:float=25.5,
                 polygon_buffer:int=0, tolerance:float=1., transform: Optional[rasterio.Affine]=None) -> List[Polygon]:
    """
    Vectorize a binary mask excluding polygons smaller than `min_area` pixels.

    Args:
        binary_water_mask: (H, W) binary mask to rasterise
        min_area: polygons with pixel area lower than this will be filtered
        polygon_buffer: buffering of the polygons
        tolerance: to simplify the polygons, in pixel units
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