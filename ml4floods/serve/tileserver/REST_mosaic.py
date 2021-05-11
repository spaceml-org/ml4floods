"""
A python class to create a mosaic from the GEE REST API (https://developers.google.com/earth-engine/apidocs) efficiently mosaicing using the Sentinel-2 utm tile geometries.
"""

import sys, os, json, io, logging
from datetime import datetime as dt
from datetime import timedelta
import urllib
logging.basicConfig(level=logging.INFO)
import multiprocessing as mp

import geopandas as gpd
from shapely import geometry, ops
from shapely.affinity import affine_transform
from PIL import Image, ImageDraw
import pyproj
import numpy as np
import rasterio
from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account
import matplotlib.pyplot as plt

from ml4floods.data import utils



def get_utm_zone(lat,lon):
    """A function to grab the UTM zone number for any lat/lon location
    """
    zone_str = str(int((lon + 180)/6) + 1)

    if ((lat>=56.) & (lat<64.) & (lon >=3.) & (lon <12.)):
        zone_str = '32'
    elif ((lat >= 72.) & (lat <84.)):
        if ((lon >=0.) & (lon<9.)):
            zone_str = '31'
        elif ((lon >=9.) & (lon<21.)):
            zone_str = '33'
        elif ((lon >=21.) & (lon<33.)):
            zone_str = '35'
        elif ((lon >=33.) & (lon<42.)):
            zone_str = '37'

    return zone_str

def _get_GEE_ids(session,start_date,end_date, aoi_wgs):
    project = 'projects/earthengine-public'
    asset_id = 'COPERNICUS/S2'
    name = '{}/assets/{}'.format(project, asset_id)
    url = 'https://earthengine.googleapis.com/v1alpha/{}:listImages?{}'.format(
      name, urllib.parse.urlencode({
        'startTime': start_date.isoformat()+'.000Z',
        'endTime': end_date.isoformat()+'.000Z',
        'region': json.dumps(geometry.mapping(aoi_wgs)),
        'filter': 'CLOUDY_PIXEL_PERCENTAGE < 99',
    }))

    response = session.get(url)
    content = response.content

    try:
        ids = [asset['id'] for asset in json.loads(content)['images']]
        cloud_coverages = [image['properties']['CLOUD_COVERAGE_ASSESSMENT'] for image in json.loads(content)['images']]
        geometries = [image['geometry'] for image in json.loads(content)['images']]
        return ids, cloud_coverages, geometries
    except:
        print('ERROR!')
        print (content)
        raise

def _get_GEE_arr(session, name, bands, x_off, y_off, patch_size, crs_code):
    url = 'https://earthengine.googleapis.com/v1alpha/{}:getPixels'.format(name)
    body = json.dumps({
        'fileFormat': 'NPY',
        'bandIds': bands,
        'grid': {
            'affineTransform': {
                'scaleX': 10,
                'scaleY': -10,
                'translateX': x_off,
                'translateY': y_off,
            },
            'dimensions': {'width': patch_size, 'height': patch_size}, #
            'crsCode': crs_code
        },
    })

    pixels_response = session.post(url, body)
    pixels_content = pixels_response.content

    try:
        arr =  np.load(io.BytesIO(pixels_content))

        return np.dstack([arr[el] for el in arr.dtype.names]).astype(np.int16)
    except:
        print ('ERROR!')
        print (pixels_content)
        raise
        
# define a worker for multiprocessing
def _fill_worker(args, google_key_path, logger):
    
    credentials = service_account.Credentials.from_service_account_file(google_key_path)
    scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/cloud-platform'])
    session = AuthorizedSession(scoped_credentials)
    
    if logger !=None:
        logger.info('created session')
    
    results = []
    for ii,ii_x,ii_y,w_NAME,w_BANDS,w_x_off,w_y_off,w_PATCH_SIZE,w_UTM_EPSG in args:
        logger.info(f'retrieving {ii},{ii_x},{ii_y}')

        arr = _get_GEE_arr(
                        session=session, 
                        name=w_NAME, 
                        bands=w_BANDS, 
                        x_off=w_x_off, 
                        y_off=w_y_off, 
                        patch_size=w_PATCH_SIZE,
                        crs_code=w_UTM_EPSG
                    )
        results.append((ii, ii_x, ii_y, arr))
        
    return results

class RESTMosaic:
    
    def __init__(self,bands = ['B4','B3','B2'], patch_size=256, scale=10,days_offset=20, workers=1, save_path='./tmp.tif',s2_tiles_path=None,google_key_path=None,verbose=False):
        # start the authorized session, load the S2_tiles, and set the parameters
        
        # set the parameters
        self.PATCH_SIZE=patch_size
        self.SCALE=scale
        self.days_offset=days_offset
        self.verbose=verbose
        self.BANDS=bands
        self.save_path=save_path
        self.workers=workers

        self.logger = logging.getLogger('RESTMosaic')
        
        # load the S2 UTM tile geometries

        if s2_tiles_path:
            if self.verbose:
                self.logger.info('Loading S2 tile geometries')
            self.s2_tiles = gpd.read_file(os.path.join(os.getcwd(),'assets','S2_tiles.gpkg')).set_index('Name')
        else:
            if self.verbose:
                self.logger.info('Get S2 tile geometries from cloud')
            self.s2_tiles = None
            
        # initialise google session
        if not google_key_path:
            google_key_path = os.path.join(os.getcwd(),'gcp-credentials.json')
        self.google_key_path=google_key_path
            
        credentials = service_account.Credentials.from_service_account_file(google_key_path)
        scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/cloud-platform'])
        self.session = AuthorizedSession(scoped_credentials)
        
        # test the session
        url = 'https://earthengine.googleapis.com/v1alpha/projects/earthengine-public/assets/LANDSAT'
        response = self.session.get(url)

        assert json.loads(response.content)['id']=='LANDSAT', 'Error with Google Session!'
        
        if self.verbose:
            self.logger.info('Successfully started google REST session')
            
    
    def mosaic(self, aoi_wgs, start_date, end_date=None, cloud_dest=None):
        
        if cloud_dest:
            self.cloud_dest=cloud_dest
        
        # get utm repoj and affine_transform for aoi
        if self.verbose:
            self.logger.info('Getting UTM reprojection data')
        self.aoi_utm, self.GT, self.Dx, self.Dy, self.reproject, self.reverse_proj, self.UTM_EPSG = self._get_proj_and_affine(aoi_wgs)
        
        # get ids
        if self.verbose:
            self.logger.info('Getting S2 ids')
        if end_date==None:
            end_date = start_date+timedelta(days=self.days_offset)
            
        _ids, cloud_coverages, geometries = _get_GEE_ids(self.session,start_date,end_date, geometry.box(*aoi_wgs.exterior.bounds))
        self.image_geometries = dict(zip(_ids,geometries))
        _ids = [x for _,x in sorted(zip(cloud_coverages,_ids))]
        
        self.image_ids = _ids


        # make two arrays for px_ix and px_iy
        if self.verbose:
            self.logger.info(f'Found {len(self.image_ids)} images. Making pixel indexer')
        self.px_x, self.px_y = self._make_pixel_indexer(self.Dx, self.Dy)
        
        # sum forward through alpha stack to get max available 
        if self.verbose:
            self.logger.info('Filling by-image mask')
        self.mask_arr, self.fills = self._mask_by_id()
        
        # if there's a single scene with more than 95% availability, just use this.
        if False:
            #if (np.array(self.fills)>0.95).sum()>=1:
            do_id = self.image_ids[np.where(np.array(self.fills)>0.95)[0].min()]
            if self.verbose:
                self.logger.info(f'found image with >95% fill, using id: {do_id}. Filling image.')
                
            self.im_arr = self._fill_single_id(do_id)
            
        else:
            if self.verbose:
                self.logger.info(f'calling mosaic. Filling Alpha mask.')

            # ... else make an alpha stack to call for px x py x N_images, subtract the earlier layers to get the alpha mask for each image
            self.alpha_mask = self._make_alpha_mask(self.Dx, self.Dy)

            # reduce each image with the pixel map to get the (ix, iy, _id)s to call
            run_idx = {}
            for ii,_id in enumerate(self.image_ids):
                run_idx[_id] = list(set(list(zip(self.px_x[self.alpha_mask[:,:,ii]>0].tolist(), self.px_y[self.alpha_mask[:,:,ii]>0].tolist()))))
            self.run_idx = run_idx

            # run forward through each image [ixs, iys], subtracting from master alpha
            if self.verbose:
                self.logger.info(f'Filling mosaic')
            self.im_arr = self._fill_mosaic()
            
        ## do output raster stuff
        self._write_rio()
        
        if self.cloud_dest:
            self.logger.info(f'to cloud: {self.save_path} -> {self.cloud_dest}')
            utils.save_file_to_bucket(self.cloud_dest, self.save_path)
            
        
    def _get_proj_and_affine(self,aoi_wgs):
        
        utm_zone = get_utm_zone(aoi_wgs.centroid.y, aoi_wgs.centroid.x)
        if aoi_wgs.centroid.y>0:
            UTM_EPSG = f'EPSG:{str(326)+str(utm_zone)}'
        else:
            UTM_EPSG = f'EPSG:{str(327)+str(utm_zone)}'
            
        proj_wgs84 = pyproj.CRS('EPSG:4326')
        proj_utm = pyproj.CRS(UTM_EPSG)
        
        reproject = pyproj.Transformer.from_crs(proj_wgs84, proj_utm, always_xy=True).transform
        reverse_proj = pyproj.Transformer.from_crs(proj_utm,proj_wgs84, always_xy=True).transform
        
        aoi_utm = ops.transform(reproject,aoi_wgs)
        
        Dx = int((aoi_utm.bounds[2]-aoi_utm.bounds[0])//self.SCALE)
        Dy = int((aoi_utm.bounds[3]-aoi_utm.bounds[1])//self.SCALE)
        
        a=e=1/self.SCALE
        b=d=0
        x_off= -aoi_utm.bounds[0]/self.SCALE
        y_off = -aoi_utm.bounds[1]/self.SCALE
        GT = [a,b,d,e,x_off,y_off]
        
        return aoi_utm, GT, Dx, Dy, reproject, reverse_proj, UTM_EPSG
        
    def _make_pixel_indexer(self,Dx, Dy):
        px_x = np.zeros((Dy,Dx))
        px_y = np.zeros((Dy,Dx))
        for idx in range(Dx//self.PATCH_SIZE+1):
            px_x[:,idx*self.PATCH_SIZE:(idx+1)*self.PATCH_SIZE] = idx
        for idy in range(Dy//self.PATCH_SIZE+1):
            px_y[idy*self.PATCH_SIZE:(idy+1)*self.PATCH_SIZE,:] = idy
            
        px_x = px_x.astype(int)
        px_y = px_y.astype(int)
    
        return px_x, px_y
    
    def _mask_by_id(self):
        mask_arr = np.zeros((self.Dy,self.Dx,len(self.image_ids))) 

        fills = []
        for ii,_id in enumerate(self.image_ids):
            if type(self.s2_tiles)==gpd.GeoDataFrame:
                granule = self.s2_tiles.loc[_id[-5:],'geometry'] # get the granule from the geodataframe
                granule = ops.unary_union(granule) # reduce geometries
                granule = granule.intersection(geometry.shape(self.image_geometries[_id])) # get intersection with image pixels
                granule_utm = ops.transform(self.reproject,granule) # flatten and transform the granule to utm
                granule_utm = ops.transform(lambda x, y, z=None: (x, y), granule_utm) # remove the z coordinates
            else:
                granule_ft = utils.load_json_from_bucket('ml4floods',f'worldfloods/s2_tiles/{_id[-5:]}.json')
                granule = geometry.shape(granule_ft['geometry'])
                granule = granule.intersection(geometry.shape(self.image_geometries[_id])) # get intersection with image pixels
                granule_utm = ops.transform(self.reproject,granule) # transform the granule to utm
                
            granule_px = affine_transform(granule_utm,self.GT) # get the granule shape into the pixel coordinates

            im = Image.fromarray(np.zeros((mask_arr.shape[0], mask_arr.shape[1])),mode='L')
            draw=ImageDraw.Draw(im)
            draw.polygon(list(granule_px.exterior.coords),fill=1)
            mask_arr[...,ii] = np.flip(np.array(im), axis=0) # flip y axis to fill from bottom
            fills.append(mask_arr[...,ii].sum()/mask_arr.shape[0]/mask_arr.shape[1])

        return mask_arr, fills
    
    def _fill_single_id(self, _id):
        # fill the mosaic
        im_arr = np.ones((self.Dy,self.Dx,len(self.BANDS)), dtype=np.int16)*-1
        for ii_x in range(self.Dx//self.PATCH_SIZE+1):
            for ii_y in range(self.Dy//self.PATCH_SIZE+1):

                NAME = 'projects/earthengine-public/assets/'+_id

                x_off = self.aoi_utm.bounds[0] + ii_x*self.PATCH_SIZE*self.SCALE
                y_off = self.aoi_utm.bounds[3] - ii_y*self.PATCH_SIZE*self.SCALE

                arr = _get_GEE_arr(
                            session=self.session, 
                            name=NAME, 
                            bands=self.BANDS, 
                            x_off=x_off, 
                            y_off=y_off, 
                            patch_size=self.PATCH_SIZE,
                            crs_code=self.UTM_EPSG
                        )
                
                if self.verbose:
                    self.logger.info(f'Getting {_id} {ii_x} {ii_y}')

                im_arr[ii_y*self.PATCH_SIZE:(ii_y+1)*self.PATCH_SIZE,ii_x*self.PATCH_SIZE:(ii_x+1)*self.PATCH_SIZE,:]=arr[:im_arr.shape[0]-ii_y*self.PATCH_SIZE,:im_arr.shape[1]-ii_x*self.PATCH_SIZE,:]

        return im_arr


    def _make_alpha_mask(self,Dx,Dy):
        main_mask = np.ones((Dy,Dx))   # a mask to iteratively mutate
        alpha_mask = np.zeros((Dy,Dx,len(self.image_ids)))  # the main alpha mask to determine api calls
        for ii,_id in enumerate(self.image_ids):

            if ii==0: # first mask do nothing
                alpha_mask[...,ii] = self.mask_arr[...,ii]
                main_mask -= self.mask_arr[...,ii]
            elif ii>0:
                alpha_mask[...,ii] = (self.mask_arr[...,ii] - self.mask_arr[...,:ii].sum(axis=-1)).clip(0,1)  # do these pixels for this image
                main_mask -= alpha_mask[...,ii] 
                
        return alpha_mask

        
    def _fill_mosaic(self):
        im_arr = np.ones((self.Dy,self.Dx,len(self.BANDS)), np.int16)*-1
        
        if self.workers==1:
            
            for ii, (_id, vv) in enumerate(self.run_idx.items()):
                for ii_x, ii_y in vv:

                    NAME = 'projects/earthengine-public/assets/'+_id

                    x_off = self.aoi_utm.bounds[0] + ii_x*self.PATCH_SIZE*self.SCALE
                    y_off = self.aoi_utm.bounds[3] - ii_y*self.PATCH_SIZE*self.SCALE

                    arr = _get_GEE_arr(
                                session=self.session, 
                                name=NAME, 
                                bands=self.BANDS, 
                                x_off=x_off, 
                                y_off=y_off, 
                                patch_size=self.PATCH_SIZE,
                                crs_code=self.UTM_EPSG
                            )

                    if self.verbose:
                        self.logger.info(f'Getting {_id} {ii_x} {ii_y}')

                    im_mask = self.alpha_mask[ii_y*self.PATCH_SIZE:(ii_y+1)*self.PATCH_SIZE,ii_x*self.PATCH_SIZE:(ii_x+1)*self.PATCH_SIZE,ii]>0

                    im_arr[ii_y*self.PATCH_SIZE:(ii_y+1)*self.PATCH_SIZE,ii_x*self.PATCH_SIZE:(ii_x+1)*self.PATCH_SIZE,:][im_mask,:]=arr[:im_arr.shape[0]-ii_y*self.PATCH_SIZE,:im_arr.shape[1]-ii_x*self.PATCH_SIZE,:][im_mask,:]
                    
        elif self.workers>1:
        
            
            args = []
            for ii, (_id, vv) in enumerate(self.run_idx.items()):
                for ii_x, ii_y in vv:

                    NAME = 'projects/earthengine-public/assets/'+_id

                    x_off = self.aoi_utm.bounds[0] + ii_x*self.PATCH_SIZE*self.SCALE
                    y_off = self.aoi_utm.bounds[3] - ii_y*self.PATCH_SIZE*self.SCALE
                    
                    args.append((ii,ii_x,ii_y,NAME, self.BANDS, x_off, y_off, self.PATCH_SIZE, self.UTM_EPSG))
            
            if self.verbose:
                self.logger.info(f'Calling REST API for {len(args)} patches with {self.workers} workers')
                
            CHUNK = len(args)//self.workers +1
            if self.verbose:
                loggers = {ii_w:logging.getLogger(str(ii_w)) for ii_w in range(self.workers)}
            else:
                loggers = {ii_w:None for ii_w in range(self.workers)}
                
            chunk_args = [
                (
                    args[ii_w*CHUNK:(ii_w+1)*CHUNK],
                    self.google_key_path,
                    loggers[ii_w]
                )
            for ii_w in range(self.workers)]
            
            pool = mp.Pool(self.workers)
            results = pool.starmap(_fill_worker, chunk_args)
            results = [item for sublist in results for item in sublist] # flatten list
            
            for ii, ii_x, ii_y, arr in results:
                
                im_mask = self.alpha_mask[ii_y*self.PATCH_SIZE:(ii_y+1)*self.PATCH_SIZE,ii_x*self.PATCH_SIZE:(ii_x+1)*self.PATCH_SIZE,ii]>0

                im_arr[ii_y*self.PATCH_SIZE:(ii_y+1)*self.PATCH_SIZE,ii_x*self.PATCH_SIZE:(ii_x+1)*self.PATCH_SIZE,:][im_mask,:]=arr[:im_arr.shape[0]-ii_y*self.PATCH_SIZE,:im_arr.shape[1]-ii_x*self.PATCH_SIZE,:][im_mask,:]                   
        
                
        return im_arr
        
    def _write_rio(self):
        
        with rasterio.open(
            os.path.join(self.save_path), 
            'w', 
            driver='COG', 
            width=self.im_arr.shape[1], 
            height=self.im_arr.shape[0], 
            count=len(self.BANDS),
            dtype=self.im_arr.dtype, 
            crs=rasterio.crs.CRS.from_string(self.UTM_EPSG), 
            transform=rasterio.transform.from_origin(west=self.aoi_utm.bounds[0], north=self.aoi_utm.bounds[3], xsize=self.SCALE, ysize=self.SCALE)) as dst:

            dst.write(np.transpose(self.im_arr,[2,0,1]), indexes=range(1,len(self.BANDS)+1))   #rast io goes channels first

    
    
    
if __name__=="__main__":
    start_date = dt(2020,7,28,0,0)
    aoi_wgs = geometry.Polygon([[-74.53043581870358, 46.03607242419274],
          [-74.35328127768796, 46.03559574915243],
          [-74.3560278597192, 46.14036520999747],
          [-74.51429964926999, 46.1370347012912]])
    
    mosaicer = RESTMosaic(verbose=True)
    mosaicer.mosaic(aoi_wgs,start_date)