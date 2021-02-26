import time, requests, logging, io, os
import numpy as np
from PIL import Image
import mercantile
import rasterio
import rasterio.warp as warp
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TEST-MODELSERVER')

from src.serve import ModelServer

logger.info('initialising modelserver object')
server = ModelServer()

logger.info('starting modelserver')
PORT=8001
server.serve(workers=4,port=PORT)


### get some array data
event_name = "EMSR333_02PORTOPALO_DEL_MONIT01_v1_observed_event_a"
z, x, y = 15, 17557, 12680     # zoom, tile_x (lon), tile_y (lat)
bounds_wgs = mercantile.bounds(int(x),int(y),int(z))    
bounds_arr = [bounds_wgs.west, bounds_wgs.south, bounds_wgs.east, bounds_wgs.north]
image_address = f"gs://worldfloods/tiffimages/S2/{event_name}.tif"
rst = rasterio.open(image_address,'r')
rast_bounds = warp.transform_bounds(rst.crs,{"init": "epsg:4326"},  *bounds_arr)
window = rasterio.windows.from_bounds(*bounds_arr, rst.transform)
OUTPUT_SHAPE=(13,256,256)

### define the seg mask colors
col_dict = {0:[0,0,0],1:[0,255,0],2:[0,0,255],3:[255,0,0]} # 0: nodata; 1: land; 2: water; 3: cloud

#arr = np.random.rand(13,256,256).astype(np.float32)
arr = rst.read(list(range(1,14)), window=window, out_shape=OUTPUT_SHAPE, boundless=True, fill_value=0).astype(np.float32)
Image.fromarray(((np.transpose(arr[(3,2,1),:,:],[1,2,0])/3000).clip(0,1)*255).astype(np.uint8)).save(os.path.join(os.getcwd(),'input_rgb.png'))
    
data = arr.tobytes()

logger.info('testing server')
for model in ['linear','unet','simplecnn']:
    
    
    logger.info(f'Calling model server for {model}...')
    tic = time.time()
    r = requests.post(f'http://127.0.0.1:{PORT}/{model}/predict', data=data, headers={'Content-Type':'application/octet-stream'})
    
    logger.info(f'Retrieved result for {model} in {time.time()-tic}s')
    recon_arr = np.load(io.BytesIO(r.content))['Y_h'] + 1 ##offset for no-data
    
    logger.info(f'reconstruction array: {recon_arr.shape}, {recon_arr.min()}, {recon_arr.max()}')
    recon_arr = recon_arr.astype(np.uint8)
    
    # Make a blank np array of zeros
    im_arr = np.zeros((256,256,3))

    # Loop through the color dictionary and project in the RGB legend color
    for kk,vv in col_dict.items():
        im_arr[recon_arr==kk,:] = np.array(vv)
    
    fname = os.path.join(os.getcwd(),f'{model}_test.png')
    Image.fromarray(im_arr.astype(np.uint8)).save(fname)
    logger.info(f'written to {fname}')

logger.info('stopping server')
server.stop()


### This also works and has a header.
#buf = io.BytesIO()
#buf.write(r.content)
#recon_arr = np.frombuffer(buf.getbuffer(), dtype=np.uint8)
