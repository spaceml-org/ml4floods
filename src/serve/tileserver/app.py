import os, mercantile, time, sys
import io
from signal import SIGTERM
from PIL import Image
import numpy as np
import rasterio
import rasterio.warp as warp
from flask import Flask, jsonify, url_for, send_file
app = Flask(__name__)


@app.route("/")
def hello():
    """ A <hello world> route to test server functionality. """
    return "Hello World TileServer!"

@app.route("/<layer_name>/<image_name>/<z>/<x>/<y>.jpg")
def servexyz(layer_name,image_name,z,x,y):
    """
    A route to get an RGB JPEG clipped from a given geotiff Sentinel-2 image for a given z,x,y TMS tile coordinate.
    Parameters
    ----------
        layer_name : str
            The layer name to serve, one of [null,mask,s2].
        image_name : str
            Image name corresponding to the geotiff (without extension) to be fetched.
        z : int
            The zoom component of a TMS tile coordinate.
        x : int
            The x component of a TMS tile coordinate.
        y : int
            The y component of a TMS tile coordinate.
    Returns
    -------
        buf (bytes): Bytestring for JPEG image
    """
    
    if layer_name=='null':
        return send_file(os.path.join(os.getcwd(),'src','serve','tileserver','static','border.png')) 
    
    try:
  
        ### get latlon bounding box from z,x,y tile request
        bounds_wgs = mercantile.bounds(int(x),int(y),int(z))  
        
        bounds_arr = [bounds_wgs.west, bounds_wgs.south, bounds_wgs.east, bounds_wgs.north]

        ################################################
        ### BY OTHERS: GET THE CLOUD PATH OF A COG TIFF
        # ... for now hardcode a single image
        #image_name = "EMSR333_02PORTOPALO_DEL_MONIT01_v1_observed_event_a"
        if layer_name=='s2':
            # This output shape is dictated by the tileserver. Using 256 is default.
            OUTPUT_SHAPE=(3,256,256)
            READBANDS=[4,3,2]
            image_address = f"gs://worldfloods/tiffimages/S2/{image_name}.tif"
        elif layer_name=='mask':
            OUTPUT_SHAPE=(1,256,256)
            READBANDS=[1]
            image_address = f"gs://worldfloods/tiffimages/gt/{image_name}.tif"
        else:
            raise

        ################################################

        # open the raster using a VRT with rasterio
        with rasterio.open(image_address,'r') as rst:

            # get the bounds for the patch in the raster crs. (will be the same is rst.crs==epsg4326)
            rast_bounds = warp.transform_bounds(rst.crs,{"init": "epsg:4326"},  *bounds_arr)

            # get the pixel slice window
            window = rasterio.windows.from_bounds(*rast_bounds, rst.transform)
            
            ## if out-of-range, return empty
            if (window.col_off<=-window.width) or ((window.col_off-rst.shape[0])>=window.width) or (window.row_off<=-window.height) or ((window.row_off-rst.shape[1])>=window.height) :
                print ('out of range')
                return send_file(os.path.join(os.getcwd(),'static','border.png')) 

            # use rasterio to read the pixel window from the cloud bucket
            rst_arr = rst.read(READBANDS, window=window, out_shape=valid_shape, boundless=True, fill_value=0)


        ####################################
        ### BY OTHERS: DO SOME PROCESSING, INFERENCE, ETC.
        ### Add other bands above
        
        if layer_name=='s2':
            ### can drive this visualisation from a SPEC in the future
            im_rgb = (np.clip(im_arr / 2500, 0, 1).transpose((1, 2, 0)) * 255).astype(np.uint8)
        elif layer_name=='mask':
            # Make a blank np array of zeros
            im_rgb = np.zeros((256,256,3))

            # Define a color legend for the flood segmentation
            col_dict = {0:[0,0,0],1:[0,255,0],2:[0,0,255],3:[255,0,0]} # 0: nodata; 1: land; 2: water; 3: cloud

            # Loop through the color dictionary and project in the RGB legend color
            for kk,vv in col_dict.items():
                im_rgb[np.squeeze(rst_arr)==kk,:] = np.array(vv)
            
        # np_arr = model(np_arr, SPEC) # <- some spec document that describes the modelling pipeline, etc. so we know how to visulise it
        ####################################
        
        buf = io.BytesIO()
        Image.fromarray(im_rgb, mode="RGB").save(buf, format="JPEG")

        return send_file(
            buf,
            as_attachment = True,
            attachment_filename = f'{z}_{x}_{y}.jpg',
            mimetype = 'image/jpeg'
        )

    except Exception as e:
            print ('ERROR!',e)
            return send_file(os.path.join(os.getcwd(),'src','serve','tileserver','static','border.png')) 

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)