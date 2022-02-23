import rasterio
import rasterio.rio.overview
import rasterio.shutil as rasterio_shutil
import os
import tempfile
import numpy as np


from typing import Optional, List

def add_overviews(rst_out, tile_size, verbose=False):
    """ Add overviews to be a cog and be displayed nicely in GIS software """

    overview_level = rasterio.rio.overview.get_maximum_overview_level(*rst_out.shape, tile_size)
    overviews = [2 ** j for j in range(1, overview_level + 1)]

    if verbose:
        print(f"Adding pyramid overviews to raster {overviews}")

    # Copied from https://github.com/cogeotiff/rio-cogeo/blob/master/rio_cogeo/cogeo.py#L274
    rst_out.build_overviews(overviews, rasterio.warp.Resampling.average)
    rst_out.update_tags(ns='rio_overview', resampling='nearest')
    tags = rst_out.tags()
    tags.update(OVR_RESAMPLING_ALG="NEAREST")
    rst_out.update_tags(**tags)
    rst_out._set_all_scales([rst_out.scales[b - 1] for b in rst_out.indexes])
    rst_out._set_all_offsets([rst_out.offsets[b - 1] for b in rst_out.indexes])


def save_cog(out_np: np.ndarray, path_tiff_save: str, profile: dict,
             descriptions:Optional[List[str]] = None,
             tags: Optional[dict] = None,
             dir_tmpfiles:str="."):
    """
    Saves `out_np` np array as a COG GeoTIFF in path_tiff_save. profile is a dict with the geospatial info to be saved
    with the TiFF.

    Args:
        out_np: 3D numpy array to save in CHW format
        path_tiff_save:
        profile: dict with profile to write geospatial info of the dataset: (crs, transform)
        descriptions: List[str]
        tags: extra dict to save as tags
        dir_tmpfiles: dir to create tempfiles if needed

    Returns:
        None

    Examples:
        >> img = np.random.randn(4,256,256)
        >> transform = rasterio.Affine(10, 0, 799980.0, 0, -10, 1900020.0)
        >> save_cog(img, "example.tif", {"crs": {"init": "epsg:32644"}, "transform":transform})
    """

    assert len(out_np.shape) == 3, f"Expected 3d tensor found tensor with shape {out_np.shape}"
    if descriptions is not None:
        assert len(descriptions) == out_np.shape[0], f"Unexpected band descriptions {len(descriptions)} expected {out_np.shape[0]}"

    # Set count, height, width
    for idx, c in enumerate(["count", "height", "width"]):
        if c in profile:
            assert profile[c] == out_np.shape[idx], f"Unexpected shape: {profile[c]} {out_np.shape}"
        else:
            profile[c] = out_np.shape[idx]

    for field in ["crs", "transform"]:
        assert field in profile, f"{field} not in profile: {profile}. it will not write cog without geo information"

    profile["BIGTIFF"] = "IF_SAFER"
    if "dtype" not in profile:
        profile["dtype"] = str(out_np.dtype)
    
    with rasterio.Env() as env:
        cog_driver = "COG" in env.drivers()

    if "RESAMPLING" not in profile:
        profile["RESAMPLING"] = "CUBICSPLINE"  # for pyramids

    if cog_driver:
        # Save tiff locally and copy it to GCP with fsspec is path is a GCP path
        if path_tiff_save.startswith("gs://"):
            import fsspec
            with tempfile.NamedTemporaryFile(dir=dir_tmpfiles, suffix=".tif", delete=True) as fileobj:
                name_save = fileobj.name
        else:
            name_save = path_tiff_save
        profile["driver"] = "COG"
        with rasterio.open(name_save, "w", **profile) as rst_out:
            if tags is not None:
                rst_out.update_tags(**tags)
            rst_out.write(out_np)
            if descriptions is not None:
                for i in range(1, out_np.shape[0] + 1):
                    rst_out.set_band_description(i, descriptions[i-1])

        if path_tiff_save.startswith("gs://"):
            fs = fsspec.filesystem("gs", requester_pays=True)
            fs.put_file(name_save, path_tiff_save)
            # subprocess.run(["gsutil", "-m", "mv", name_save, path_tiff_save])
            if os.path.exists(name_save):
                os.remove(name_save)

        return path_tiff_save

    print("COG driver not available. Generate COG manually with GTiff driver")
    # If COG driver is not available (GDAL < 3.1) we go to copying the file using GTiff driver
    # Set blockysize, blockxsize
    for idx, b in enumerate(["blockysize", "blockxsize"]):
        if b in profile:
            assert profile[b] <= 512, f"{b} is {profile[b]} must be <=512 to be displayed in GEE "
        else:
            profile[b] = min(512, out_np.shape[idx + 1])

    if (out_np.shape[1] >= 512) or (out_np.shape[2] >= 512):
        profile["tiled"] = True

    profile["driver"] = "GTiff"
    with tempfile.NamedTemporaryFile(dir=dir_tmpfiles, suffix=".tif", delete=True) as fileobj:
        named_tempfile = fileobj.name

    with rasterio.open(named_tempfile, "w", **profile) as rst_out:
        if tags is not None:
            rst_out.update_tags(**tags)
        rst_out.write(out_np)
        if descriptions is not None:
            for i in range(1, out_np.shape[0] + 1):
                rst_out.set_band_description(i, descriptions[i - 1])
        
        add_overviews(rst_out, tile_size=profile["blockysize"])
        print("Copying temp file")
        rasterio_shutil.copy(rst_out, path_tiff_save, copy_src_overviews=True, tiled=True,
                             blockxsize=profile["blockxsize"],
                             blockysize=profile["blockysize"],
                             driver="GTiff")

    rasterio_shutil.delete(named_tempfile)
    return path_tiff_save
