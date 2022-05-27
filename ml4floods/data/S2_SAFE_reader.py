"""
Sentinel-2 reader inherited from https://github.com/IPL-UV/DL-L8S2-UV.

It has several enhancements:
* Support for S2L2A images
* It can read directly images from a GCP bucket (for example data from  [here](https://cloud.google.com/storage/docs/public-datasets/sentinel-2))
* Windowed read and read and reproject in the same function (see `load_bands_bbox`)
* Creation of the image only involves reading one metadata file (`xxx.SAFE/MTD_{self.producttype}.xml`)


https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/document-library

"""
from rasterio import windows, features, coords
from rasterio.warp import reproject
from shapely.geometry import Polygon, MultiPolygon
from xml.etree.ElementTree import parse
import xml.etree.ElementTree as ET
import rasterio
import datetime
import io
from collections import OrderedDict
import numbers
import numpy as np
import os
import re
import tempfile
import sys
from typing import List, Tuple, Union


BANDS_S2 = ["B01", "B02","B03", "B04", "B05", "B06",
            "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]

BANDS_RESOLUTION = OrderedDict({"B01": 60, "B02": 10,
                                "B03": 10, "B04": 10,
                                "B05": 20, "B06": 20,
                                "B07": 20, "B08": 10,
                                "B8A": 20, "B09": 60,
                                "B10": 60, "B11": 20, "B12": 20})


def process_metadata_msi(xml_file:str) -> Tuple[List[str], Polygon]:
    """
    Read the xml metadata and
    1) It gets the paths of the jp2 files of the SAFE product. It excludes the _TCI files.
    2) It gets the footprint of the image (polygon where image has valid values)
    Params:
        xml_file: description file found in root of .SAFE folder.  xxxx.SAFE/MTD_MSIL1C.xml or xxxx.SAFE/MTD_MSIL2A.xml

    Returns:
        List of jp2 files in the xml file
        Polygon with the extent of the image
    """
    if xml_file.startswith("gs://"):
        import fsspec
        fs = fsspec.filesystem("gs",requester_pays=True)
        with fs.open(xml_file, "rb") as file_obj:
            root = ET.fromstring(file_obj.read())
    else:
        root = ET.parse(xml_file).getroot()

    bands_elms = root.findall(".//IMAGE_FILE")
    jp2bands = [b.text+".jp2" for b in bands_elms if not b.text.endswith("_TCI")]

    footprint_txt = root.findall(".//EXT_POS_LIST")[0].text
    coords_split = footprint_txt.split(" ")[:-1]
    footprint = Polygon([(float(lngstr), float(latstr)) for latstr, lngstr in zip(coords_split[::2], coords_split[1::2])])

    return jp2bands, footprint


class S2Image:
    def __init__(self, s2_folder, out_res=10):
        mission, self.producttype, sensing_date_str, pdgs, relorbitnum, tile_number_field, product_discriminator = s2_name_split(
            s2_folder)

        out_res = int(out_res)
        assert out_res in {10, 20, 60}, "Not valid output resolution.Choose 10, 20, 60"

        # Default resolution to read
        self.out_res = out_res

        # Remove last trailing slash
        s2_folder = s2_folder[:-1] if s2_folder.endswith("/") else s2_folder
        self.name = os.path.basename(os.path.splitext(s2_folder)[0])

        self.datetime = datetime.datetime.strptime(sensing_date_str, "%Y%m%dT%H%M%S").replace(
            tzinfo=datetime.timezone.utc)
        self.folder = s2_folder
        self.metadata_msi = os.path.join(self.folder, f"MTD_{self.producttype}.xml").replace("\\","/")
        # load _pol from geometric_info product footprint!
        jp2bands, self._pol = process_metadata_msi(self.metadata_msi)

        self.granule_folder = os.path.dirname(os.path.dirname(jp2bands[0]))
        self.bands = list(BANDS_S2)

        self.band_check = None
        for band, res in BANDS_RESOLUTION.items():
            if res == self.out_res:
                self.band_check = band
                break

        self.granule = [os.path.join(self.folder, b).replace("\\","/") for b in jp2bands]

    def polygon(self):
        """ Footprint polygon of the image in lat/lng (epsg:4326) """
        return self._pol

    def _load_attrs(self):
        band_check_idx = self.bands.index(self.band_check)
        with rasterio.open(self.granule[band_check_idx], driver='JP2OpenJPEG') as src:
            self._shape = src.shape
            self._transform = src.transform
            self._crs = src.crs
            self._bounds = src.bounds
            self._dtype = src.dtypes[0]

    @property
    def dtype(self):
        if not hasattr(self, "_dtype"):
            self._load_attrs()
        return self._dtype

    @property
    def shape(self):
        if not hasattr(self, "_shape"):
            self._load_attrs()
        return self._shape

    @property
    def transform(self):
        if not hasattr(self, "_transform"):
            self._load_attrs()
        return self._transform

    @property
    def crs(self):
        if not hasattr(self, "_crs"):
            self._load_attrs()
        return self._crs

    @property
    def bounds(self):
        if not hasattr(self, "_bounds"):
            self._load_attrs()
        return self._bounds

    def __str__(self):
        return self.folder

    def load_bands_bbox(self, bbox_read, dst_crs=None, bands=None, resolution_dst_crs=None,
                        resampling=rasterio.warp.Resampling.cubic_spline) -> Tuple[np.ndarray, rasterio.Affine]:
        """
        Loads a list of bands in the bounding box provided at resolution_dst_crs

        :param bbox_read: bounding box to read in dst_crs coordinate reference system (CRS)
        :param dst_crs: dst crs e.g. {"init": "epsg:3857"}
        :param bands: id band to read from self.granule
        :param resolution_dst_crs: tuple or int with out resolution
        :param resampling: resampling method.

        :return: 3D numpy array of the reprojected data and out transform
        """
        if bands is None:
            bands = list(range(len(self.granule)))

        if dst_crs is None:
            dst_crs = self.crs
        else:
            assert resolution_dst_crs is not None, "Resolution must be set if dst_crs is set"

        if resolution_dst_crs is None:
            resolution_dst_crs = (self.out_res, self.out_res)

        dst_tuple =  read_tiffs_bbox_crs([self.granule[b] for b in bands],
                                         bbox_read, dst_crs,
                                         resolution_dst_crs=resolution_dst_crs, resampling=resampling,
                                         dtpye_dst=self.dtype)
        return dst_tuple

    def load_bands(self, bands=None, window=None):
        """
        window is a rasterio.windows.Window in self.out_res resolution

        :param bands: indexes of bands to read (self.granule[iband] for iband in bands)
        :param window: rasterio.windows.Window object
        :return: CHW array
        """
        if bands is None:
            bands = list(range(len(self.granule)))

        if window is None:
            shape = self.shape
            window = rasterio.windows.Window(col_off=0, row_off=0,
                                             width=shape[1], height=shape[0])

        # Use src.read if all bands have the same resolution == out_res
        if all(BANDS_RESOLUTION[self.bands[iband]] == self.out_res  for iband in bands):
            shape_window = windows.shape(window)
            dest_array = np.zeros((len(bands), ) + shape_window, dtype=self.dtype)
            for _i, iband in enumerate(bands):
                with rasterio.open(self.granule[iband]) as src:
                    dest_array[_i] = src.read(1, window=window, boundless=True)

            if window is not None:
                transform = rasterio.windows.transform(window, self.transform)
            else:
                transform = self.transform
            return dest_array, transform

        # Get bounding box to read
        bbox = windows.bounds(window, self.transform)
        return self.load_bands_bbox(bbox, bands=bands)

    def load_mask(self, window=None):
        band_check_idx = self.bands.index(self.band_check)
        band = self.load_bands(bands=[band_check_idx], window=window)
        return (band == 0) | (band == (2**16)-1)


class S2ImageL2A(S2Image):
    def __init__(self, s2_folder, out_res=10):
        super(S2ImageL2A, self).__init__(s2_folder, out_res=out_res)
        assert self.producttype == "MSIL2A", f"Unexpected product type {self.producttype} in image {self.folder}"

        # see https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR#bands for a description of the granules data

        self.all_granules = list(self.granule)

        # Filter bands in self.all_granules according to the out_res
        self.granule = []
        self.bands = []
        for b in BANDS_S2:
            if b == "B10":
                continue
            res_band = BANDS_RESOLUTION[b]
            res_band = res_band if out_res < res_band else out_res
            band_and_res = f"{b}_{res_band}m.jp2"
            granules_match = [g for g in self.all_granules if g.endswith(band_and_res)]
            assert len(granules_match) == 1, f"Granules that match {band_and_res}: {granules_match}"
            self.granule.append(granules_match[0])
            self.bands.append(b)

        # Use SCL band for clouds
        res_band = 20 if out_res < 20 else out_res
        band_and_res = f"SCL_{res_band}m.jp2"
        granules_match = [g for g in self.all_granules if g.endswith(band_and_res)]
        assert len(granules_match) == 1, f"Granules that match {band_and_res}: {granules_match}"
        self.slc_granule = granules_match[0]

    def load_scl_bbox(self, bbox_read, dst_crs=None,resolution_dst_crs=None):
        if dst_crs is None:
            dst_crs = self.crs
        else:
            assert resolution_dst_crs is not None, "Resolution must be set if dst_crs is set"

        if resolution_dst_crs is None:
            resolution_dst_crs = (self.out_res, self.out_res)

        slc_tuple = read_tiffs_bbox_crs([self.slc_granule],
                                        bbox_read, dst_crs, resampling=rasterio.warp.Resampling.nearest,
                                        resolution_dst_crs=resolution_dst_crs,dtpye_dst=np.uint8)

        return slc_tuple


class S2ImageL1C(S2Image):
    def __init__(self, s2_folder, out_res=10, read_metadata=False):
        super(S2ImageL1C,self).__init__(s2_folder, out_res=out_res)

        assert self.producttype == "MSIL1C", f"Unexpected product type {self.producttype} in image {self.folder}"

        self.all_granules = list(self.granule)

        self.msk_clouds_file = os.path.join(self.folder, self.granule_folder, "MSK_CLOUDS_B00.gml").replace("\\","/")
        self.metadata_tl = os.path.join(self.folder, self.granule_folder, "MTD_TL.xml").replace("\\","/")

        self.bands = list(BANDS_S2)

        if read_metadata:
            self.read_metadata()

        self.granule = self.all_granules

        assert len(self.bands) == len(self.granule), f"Expected same but found different {len(self.bands)} {len(self.granule)}"

        # Granule in L1C does not include TCI
        # Assert bands in self.granule are ordered as in BANDS_S2
        assert len(self.granule) == len(BANDS_S2), f"Unexpected number of granules {len(self.granule)} expected {len(BANDS_S2)}"
        assert all(granule[-7:-4] == bname for bname, granule in zip(BANDS_S2, self.granule)), f"some granules are not in the expected order {self.granule}"

    def read_metadata(self):
        '''
        Read metadata TILE to parse information about the acquisition and properties of GRANULE bands
        Source: fmask.sen2meta.py
        :return: relevant attributes
        '''
        with open(self.metadata_tl) as f:
            root = ET.fromstring(f.read())
            # Stoopid XML namespace prefix
            nsPrefix = root.tag[:root.tag.index('}') + 1]
            nsDict = {'n1': nsPrefix[1:-1]}

            generalInfoNode = root.find('n1:General_Info', nsDict)
            # N.B. I am still not entirely convinced that this SENSING_TIME is really
            # the acquisition time, but the documentation is rubbish.
            sensingTimeNode = generalInfoNode.find('SENSING_TIME')
            sensingTimeStr = sensingTimeNode.text.strip()
            self.datetime = datetime.datetime.strptime(sensingTimeStr, "%Y-%m-%dT%H:%M:%S.%fZ")
            tileIdNode = generalInfoNode.find('TILE_ID')
            tileIdFullStr = tileIdNode.text.strip()
            self.tileId = tileIdFullStr.split('_')[-2]
            self.satId = tileIdFullStr[:3]
            self.procLevel = tileIdFullStr[13:16]  # Not sure whether to use absolute pos or split by '_'....

            geomInfoNode = root.find('n1:Geometric_Info', nsDict)
            geocodingNode = geomInfoNode.find('Tile_Geocoding')
            epsgNode = geocodingNode.find('HORIZONTAL_CS_CODE')
            self.epsg = epsgNode.text.split(':')[1]

            # Dimensions of images at different resolutions.
            self.dimsByRes = {}
            sizeNodeList = geocodingNode.findall('Size')
            for sizeNode in sizeNodeList:
                res = sizeNode.attrib['resolution']
                nrows = int(sizeNode.find('NROWS').text)
                ncols = int(sizeNode.find('NCOLS').text)
                self.dimsByRes[res] = (nrows, ncols)

            # Upper-left corners of images at different resolutions. As far as I can
            # work out, these coords appear to be the upper left corner of the upper left
            # pixel, i.e. equivalent to GDAL's convention. This also means that they
            # are the same for the different resolutions, which is nice.
            self.ulxyByRes = {}
            posNodeList = geocodingNode.findall('Geoposition')
            for posNode in posNodeList:
                res = posNode.attrib['resolution']
                ulx = float(posNode.find('ULX').text)
                uly = float(posNode.find('ULY').text)
                self.ulxyByRes[res] = (ulx, uly)

            # Sun and satellite angles.
            tileAnglesNode = geomInfoNode.find('Tile_Angles')
            sunZenithNode = tileAnglesNode.find('Sun_Angles_Grid').find('Zenith')
            self.angleGridXres = float(sunZenithNode.find('COL_STEP').text)
            self.angleGridYres = float(sunZenithNode.find('ROW_STEP').text)
            self.sunZenithGrid = self.makeValueArray(sunZenithNode.find('Values_List'))
            sunAzimuthNode = tileAnglesNode.find('Sun_Angles_Grid').find('Azimuth')
            self.sunAzimuthGrid = self.makeValueArray(sunAzimuthNode.find('Values_List'))
            self.anglesGridShape = self.sunAzimuthGrid.shape

            # Now build up the viewing angle per grid cell, from the separate layers
            # given for each detector for each band. Initially I am going to keep
            # the bands separate, just to see how that looks.
            # The names of things in the XML suggest that these are view angles,
            # but the numbers suggest that they are angles as seen from the pixel's
            # frame of reference on the ground, i.e. they are in fact what we ultimately want.
            viewingAngleNodeList = tileAnglesNode.findall('Viewing_Incidence_Angles_Grids')
            self.viewZenithDict = self.buildViewAngleArr(viewingAngleNodeList, 'Zenith')
            self.viewAzimuthDict = self.buildViewAngleArr(viewingAngleNodeList, 'Azimuth')

            # Make a guess at the coordinates of the angle grids. These are not given
            # explicitly in the XML, and don't line up exactly with the other grids, so I am
            # making a rough estimate. Because the angles don't change rapidly across these
            # distances, it is not important if I am a bit wrong (although it would be nice
            # to be exactly correct!).
            (ulx, uly) = self.ulxyByRes["10"]
            self.anglesULXY = (ulx - self.angleGridXres / 2.0, uly + self.angleGridYres / 2.0)

    def buildViewAngleArr(self, viewingAngleNodeList, angleName):
        """
        Build up the named viewing angle array from the various detector strips given as
        separate arrays. I don't really understand this, and may need to re-write it once
        I have worked it out......

        The angleName is one of 'Zenith' or 'Azimuth'.
        Returns a dictionary of 2-d arrays, keyed by the bandId string.
        """
        angleArrDict = {}
        for viewingAngleNode in viewingAngleNodeList:
            bandId = viewingAngleNode.attrib['bandId']
            angleNode = viewingAngleNode.find(angleName)
            angleArr = self.makeValueArray(angleNode.find('Values_List'))
            if bandId not in angleArrDict:
                angleArrDict[bandId] = angleArr
            else:
                mask = (~np.isnan(angleArr))
                angleArrDict[bandId][mask] = angleArr[mask]
        return angleArrDict

    def get_polygons_bqa(self):
        def polygon_from_coords(coords, fix_geom=False, swap=True, dims=2):
            """
            Return Shapely Polygon from coordinates.
            - coords: list of alterating latitude / longitude coordinates
            - fix_geom: automatically fix geometry
            """
            assert len(coords) % dims == 0
            number_of_points = int(len(coords) / dims)
            coords_as_array = np.array(coords)
            reshaped = coords_as_array.reshape(number_of_points, dims)
            points = [(float(i[1]), float(i[0])) if swap else ((float(i[0]), float(i[1]))) for i in reshaped.tolist()]
            polygon = Polygon(points).buffer(0)
            try:
                assert polygon.is_valid
                return polygon
            except AssertionError:
                if fix_geom:
                    return polygon.buffer(0)
                else:
                    raise RuntimeError("Geometry is not valid.")


        exterior_str = str("eop:extentOf/gml:Polygon/gml:exterior/gml:LinearRing/gml:posList")
        interior_str = str("eop:extentOf/gml:Polygon/gml:interior/gml:LinearRing/gml:posList")
        root = parse(self.msk_clouds_file).getroot()
        nsmap = {k: v for k, v in root.nsmap.items() if k}
        try:
            for mask_member in root.iterfind("eop:maskMembers", namespaces=nsmap):
                for feature in mask_member:
                    type = feature.findtext("eop:maskType", namespaces=nsmap)

                    ext_elem = feature.find(exterior_str, nsmap)
                    dims = int(ext_elem.attrib.get('srsDimension', '2'))
                    ext_pts = ext_elem.text.split()
                    exterior = polygon_from_coords(ext_pts, fix_geom=True, swap=False, dims=dims)
                    try:
                        interiors = [polygon_from_coords(int_pts.text.split(), fix_geom=True, swap=False, dims=dims)
                                     for int_pts in feature.findall(interior_str, nsmap)]
                    except AttributeError:
                        interiors = []

                    yield dict(geometry=Polygon(exterior, interiors).buffer(0),
                               attributes=dict(maskType=type),
                               interiors=interiors)

        except StopIteration:
            yield dict(geometry=Polygon(),
                       attributes=dict(maskType=None),
                       interiors=[])
            raise StopIteration()

    def load_clouds_bqa(self, window=None):
        mask_types = ["OPAQUE", "CIRRUS"]
        poly_list = list(self.get_polygons_bqa())

        nrows, ncols = self.shape
        transform_ = self.transform

        def get_mask(mask_type=mask_types[0]):
            assert mask_type in mask_types, "mask type must be OPAQUE or CIRRUS"
            fill_value = {m: i+1 for i, m in enumerate(mask_types)}
            n_polys = np.sum([poly["attributes"]["maskType"] == mask_type for poly in poly_list])
            msk = np.zeros(shape=(nrows, ncols), dtype=np.float32)
            if n_polys > 0:
                # n_interiors = np.sum([len(poly) for poly in poly_list if poly["interiors"]])
                multi_polygon = MultiPolygon([poly["geometry"]
                                              for poly in poly_list
                                              if poly["attributes"]["maskType"] == mask_type]).buffer(0)
                bounds = multi_polygon.bounds
                bbox2read = coords.BoundingBox(*bounds)
                window_read = windows.from_bounds(*bbox2read, transform_)
                slice_read = tuple(slice(int(round(s.start)), int(round(s.stop))) for s in window_read.toslices())
                out_shape = tuple([s.stop - s.start for s in slice_read])
                transform_slice = windows.transform(window_read, transform_)

                shapes = [({"type": "Polygon",
                            "coordinates": [np.stack([
                                p_elem["geometry"].exterior.xy[0],
                                p_elem["geometry"].exterior.xy[1]], axis=1).tolist()]}, fill_value[mask_type])
                          for p_elem in poly_list if p_elem["attributes"]['maskType'] == mask_type]
                sub_msk = features.rasterize(shapes=shapes, fill=0,
                                             out_shape=out_shape, dtype=np.float32,
                                             transform=transform_slice)
                msk[slice_read] = sub_msk

            return msk

        if window is None:
            shape = self.shape
            window = rasterio.windows.Window(col_off=0, row_off=0,
                                             width=shape[1], height=shape[0])

        mask = self.load_mask(window=window)

        slice_ = window.toslices()

        msk_op_cirr = [np.ma.MaskedArray(get_mask(mask_type=m)[slice_], mask=mask) for m in mask_types]
        msk_clouds = np.ma.MaskedArray(np.clip(np.sum(msk_op_cirr, axis=0), 0, 1), mask=mask)
        return msk_clouds

    @staticmethod
    def makeValueArray(valuesListNode):
        """
        Take a <Values_List> node from the XML, and return an array of the values contained
        within it. This will be a 2-d numpy array of float32 values (should I pass the dtype in??)

        """
        valuesList = valuesListNode.findall('VALUES')
        vals = []
        for valNode in valuesList:
            text = valNode.text
            vals.append([np.float32(x) for x in text.strip().split()])

        return np.array(vals)


def s2loader(s2folder:str, out_res:int=10) -> Union[S2ImageL2A, S2ImageL1C]:
    """
    Loads a S2ImageL2A or S2ImageL1C depending on the product type

    :param s2folder: .SAFE folder. Expected standard ESA naming convention (see s2_name_split fun)
    :param out_res: default output resolution

    """
    _, producttype_nos2, _, _, _, _, _ = s2_name_split(s2folder)
    if producttype_nos2 == "MSIL2A":
        return S2ImageL2A(s2folder, out_res=out_res)
    elif producttype_nos2 == "MSIL1C":
        return S2ImageL1C(s2folder, out_res=out_res)

    raise NotImplementedError(f"Don't know how to load {producttype_nos2} products")


NEW_FORMAT = "(S2\w{1})_(MSIL\w{2})_(\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2})_(\w{5})_(\w{4})_T(\w{5})_(\w{15})"
OLD_FORMAT = "(S2\w{1})_(\w{4})_(\w{3}_\w{6})_(\w{4})_(\d{8}T\d{6})_(\w{4})_V(\d{4}\d{2}\d{2}T\d{6})_(\d{4}\d{2}\d{2}T\d{6})"

def s2_name_split(s2file):
    """
    https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/naming-convention

    ```
    s2l1c = "S2A_MSIL1C_20151218T182802_N0201_R127_T11SPD_20151218T182756.SAFE"
    mission, producttype, sensing_date_str, pdgs, relorbitnum, tile_number_field, product_discriminator = s2_name_split(s2l1c)
    ```

    S2A_MSIL1C_20151218T182802_N0201_R127_T11SPD_20151218T182756.SAFE
    MMM_MSIXXX_YYYYMMDDTHHMMSS_Nxxyy_ROOO_Txxxxx_<Product Discriminator>.SAFE
    MMM: is the mission ID(S2A/S2B)
    MSIXXX: MSIL1C denotes the Level-1C product level/ MSIL2A denotes the Level-2A product level
    YYYYMMDDHHMMSS: the datatake sensing start time
    Nxxyy: the PDGS Processing Baseline number (e.g. N0204)
    ROOO: Relative Orbit number (R001 - R143)
    Txxxxx: Tile Number field
    SAFE: Product Format (Standard Archive Format for Europe)

    :param s2l1c:
    :return:
    """
    s2file = s2file[:-1] if s2file.endswith("/") else s2file
    basename = os.path.basename(os.path.splitext(s2file)[0])
    matches = re.match(NEW_FORMAT, basename)
    if matches is not None:
        return matches.groups()


def s2_old_format_name_split(s2file):
    """
    https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/naming-convention

    ```
    s2l1c = "S2A_OPER_PRD_MSIL1C_PDMC_20151206T093912_R090_V20151206T043239_20151206T043239.SAFE"
    mission, opertortest, filetype, sitecenter,  creation_date_str, relorbitnum, sensing_time_start, sensing_time_stop = s2_old_format_name_split(s2l1c)
    ```

    :param s2l1c:
    :return:
    """
    s2file = s2file[:-1] if s2file.endswith("/") else s2file
    basename = os.path.basename(os.path.splitext(s2file)[0])
    matches = re.match(OLD_FORMAT, basename)
    if matches is not None:
        return matches.groups()


def pad_window(window, pad_size):
    return rasterio.windows.Window(window.col_off - pad_size[1],
                                   window.row_off - pad_size[0],
                                   width=window.width + 2*pad_size[1],
                                   height=window.height + 2*pad_size[1])

def round_window(window):
    return window.round_lengths(op="ceil").round_offsets(op="floor")


def read_tiffs_bbox_crs(tiff_files, bbox_read, dst_crs, resolution_dst_crs, resampling, dtpye_dst=np.float32):
    """
    Read data from a list of tiffs. Read the data in bbox_read reprojected to dst_crs with resolution resolution_dst_crs.

    :param tiff_files:
    :param bbox_read: Bounding box to read in dst_crs CRS
    :param dst_crs:
    :param resampling:
    :param resolution_dst_crs:
    :param dtpye_dst:
    :return:
    """

    if isinstance(resolution_dst_crs, numbers.Number):
        resolution_dst_crs = (abs(resolution_dst_crs), abs(resolution_dst_crs))

    # Compute affine transform out crs
    dst_transform = transform_bounds_res(bbox_read, resolution_dst_crs)
    # Compute size of window in out crs
    window_out = rasterio.windows.from_bounds(*bbox_read,
                                              transform=dst_transform).round_lengths(op="ceil")
    # Create out array
    destination = np.zeros((len(tiff_files), window_out.height, window_out.width),
                           dtype=dtpye_dst)

    for i, b in enumerate(tiff_files):
        try:
            np_array_in, transform_in, src_crs = read_overlap(b, bbox_read, dst_crs)
        except Exception as e:
            if b.startswith("gs://"):
                print(f"\tError reading remote file {sys.exc_info()[0]}. \n Copying file to local and trying again")
                from google.cloud import storage
                client = storage.Client()
                with tempfile.NamedTemporaryFile() as tmpfile:
                    client.download_blob_to_file(b, tmpfile)
                    np_array_in, transform_in, src_crs = read_overlap(tmpfile.name, bbox_read, dst_crs)
            else:
                raise e

        np_array_in = np_array_in.astype(dtpye_dst)

        rasterio.warp.reproject(
            np_array_in,
            destination[i],
            src_transform=transform_in,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=0.,
            dst_nodata=0.,
            resampling=resampling)

    return destination, dst_transform


def read_overlap(tiff_file, bbox_read, dst_crs, pad_add=(3, 3)):
    with rasterio.open(tiff_file) as src_b:
        bounds_in = rasterio.warp.transform_bounds(dst_crs, src_b.crs, *bbox_read)
        window_in = rasterio.windows.from_bounds(*bounds_in, transform=src_b.transform)
        window_in = pad_window(window_in, pad_add)  # Add padding for bicubic int
        window_in = round_window(window_in)
        transform_in = rasterio.windows.transform(window_in, transform=src_b.transform)

        src_crs = src_b.crs
        np_array_in = src_b.read(1, window=window_in, boundless=True)
    return np_array_in, transform_in, src_crs



def mosaic_s2(s2objs, bbox_read, dst_crs, res_s2, threshold_stop_invalid=1/1000., sort=True):
    """
    Mosaic a list of s2objs. Reorder s2objs by largest overlap

    :param s2objs: Could be L2A or L1C S2 products.
    :param bbox_read:
    :param dst_crs:
    :param res_s2:
    :param threshold_stop_invalid:
    :param sort: Sort s2ojs
    :return:
    """
    # Order objects by overlap with pol_read
    bbox_lnglat = rasterio.warp.transform_bounds(dst_crs,
                                                 {'init': 'epsg:4326'},
                                                 *bbox_read)
    pol_read = Polygon(generate_polygon(bbox_lnglat))

    if sort:
        s2objs = sorted(s2objs, key= lambda s2: s2.polygon().intersection(pol_read).area/pol_read.area,
                        reverse=True)

    invalid_destination = None
    current_clouds = None
    for s2obj in s2objs:
        print(f"{s2obj.name} reprojecting")
        destination_iter, dst_transform = s2obj.load_bands_bbox(bbox_read, dst_crs=dst_crs,
                                                                resolution_dst_crs=(res_s2, res_s2))

        if invalid_destination is None:
            destination_array = destination_iter
            if s2obj.producttype == "MSIL2A":
                current_clouds, _ = s2obj.load_scl_bbox(bbox_read, dst_crs=dst_crs,
                                                        resolution_dst_crs=(res_s2, res_s2))
                current_clouds = current_clouds[0]
        else:
            destination_array[:, invalid_destination] = destination_iter[:, invalid_destination]
            if s2obj.producttype == "MSIL2A":
                slc, _ = s2obj.load_scl_bbox(bbox_read, dst_crs=dst_crs, resolution_dst_crs=(res_s2, res_s2))
                slc = slc[0]
                current_clouds[invalid_destination] = slc[invalid_destination]

        if s2obj.producttype == "MSIL2A":
            invalid_destination = current_clouds <= 1
        else:
            invalid_destination = np.all(destination_array < 1e-6, axis=0, keepdims=False) |\
                                  np.all(destination_array == (2 ** 16) - 1, axis=0, keepdims=False)

        frac_invalids = np.sum(invalid_destination) / np.prod(invalid_destination.shape)

        if frac_invalids < threshold_stop_invalid:
            break

    return destination_array, current_clouds


def generate_polygon(bbox):
    """
    Generates a list of coordinates: [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x1,y1]]
    """
    return [[bbox[0],bbox[1]],
             [bbox[2],bbox[1]],
             [bbox[2],bbox[3]],
             [bbox[0],bbox[3]],
             [bbox[0],bbox[1]]]

def transform_bounds_res(bbox_read, resolution_dst_crs):
    """ Compute affine transform for a given bounding box and resolution. bbox_read and resolution_dst_crs are expected in the same CRS"""
    # Compute affine transform out crs
    return rasterio.transform.from_origin(min(bbox_read[0], bbox_read[2]),
                                          max(bbox_read[1], bbox_read[3]),
                                          resolution_dst_crs[0], resolution_dst_crs[1])