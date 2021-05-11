import io
import os

import mercantile
import numpy as np
import pandas as pd
import rasterio
import rasterio.windows
import rasterio.warp as warp
from flask import Flask, jsonify, send_file, request
from PIL import Image
from ml4floods.data.copernicusEMS import activations
from ml4floods.serve.tileserver import helpers
from ml4floods.serve.tileserver.ingest import Ingestor
from ml4floods.data import utils
import geopandas as gpd
import json
import geojson
from shapely import geometry
from datetime import datetime as dt
from google.cloud import storage

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


### on startup grab copernicus events -> might want to periodically rerun this
events_df = activations.table_floods_ems()
countries_df = pd.DataFrame(json.load(open(os.path.join(os.getcwd(),'ml4floods','serve','tileserver','static','countries.json'),'r')))
events_df = helpers.postprocess_floodtable(events_df, countries_df)
updated_df = helpers.walk_bucket(events_df)
base_gj = gpd.read_file(os.path.join(os.getcwd(),'assets','gj_gdf.gpkg'))
base_gj = helpers.refresh_geojson(base_gj, updated_df)

"""
if os.path.exists(os.path.join(os.getcwd(),'assets','gj_gdf.gpkg')):
    print ('found local gj_gdf')
    gj_gdf = gpd.read_file(os.path.join(os.getcwd(),'assets','gj_gdf.gpkg'))
else:
    gj_gdf = gpd.GeoDataFrame(pd.DataFrame(columns=['CODE','AOI','geometry']), geometry='geometry')
    
gj_gdf = helpers.refresh_geojson(gj_gdf, updated_df)
if not os.path.exists(os.path.join(os.getcwd(),'assets','gj_gdf.gpkg')):
    gj_gdf.to_file(os.path.join(os.getcwd(),'assets','gj_gdf.gpkg'), driver='GPKG', overwrite=True)
""" 
    
print (base_gj)


@app.route('/post/custom/', methods=['POST'])
def post_custom_aoi():
    data = request.json
    print ('custom inbound')
    print (data)
    
    ### put the custom event in meta
    ft = geojson.Feature(
        geometry=data['geometry'],
        properties={
            'event-date':data['date'],
        }
    )
    fname = f'CUSTOM-{data["submission_dt"]}_AOI01_custom.json'
    
    tmp_path = os.path.join(os.getcwd(),'tmp',fname)
    json.dump(ft,open(tmp_path,'w'))
    target_path= os.path.join('gs://ml4floods','worldfloods','lk-dev','meta',fname)
    
    utils.save_file_to_bucket(target_directory=target_path, source_directory=tmp_path)
    
    return jsonify({'status':'succcess'})


@app.route('/get/geojson/', methods=['GET'])
def get_geojson():

    updated_df = helpers.walk_bucket(events_df) # update any custom events
    gj_gdf = helpers.refresh_geojson(base_gj, updated_df)
    gj_gdf['event-date'] = gj_gdf['event-date'].dt.strftime('%Y-%m-%d')
    print ('custom events')
    print (gj_gdf.loc[gj_gdf['CODE'].str.contains('CUSTOM'),:])
    print (gj_gdf.columns)
    #gj_gdf['CODE-AOI-DATE'] = gj_gdf['CODE']+'_'+gj_gdf['AOI']+'_'+gj_gdf['event-date']
    #gj_gdf = gj_gdf.drop(columns=['CODE','event-date','AOI'])
    gj_gdf.to_file(os.path.join(os.getcwd(),'tmp','gj_gdf.geojson'),driver='GeoJSON')
    
    return send_file(os.path.join(os.getcwd(),'tmp','gj_gdf.geojson'), attachment_filename = f'geojson_{dt.now().isoformat()}.json')  
    

@app.route('/get/ingest/', methods=['GET'])
def get_runingest():
    ems_code = request.args.get('CODE')
    aoi_code = request.args.get('AOI')
    event_date = request.args.get('event-date')
    print ('ingesting aoi code:',ems_code, aoi_code)
    if 'CUSTOM' in ems_code:
        aoi_geom = request.args.get('aoi_geom')
        print ('aoi_geom',type(aoi_geom),aoi_geom)
        aoi_geom = geometry.shape(json.loads(aoi_geom))
        print ('ingesting custom AOI')

        ingestor = Ingestor(
            local_path='tmp',
            bucket='ml4floods',
            cloud_path = {
                'S2-pre':'worldfloods/lk-dev/S2-pre',
                'S2-post':'worldfloods/lk-dev/S2-post',
                'GT':'worldfloods/lk-dev/gt',
                'meta':'worldfloods/lk-dev/meta',
                'ML':'worldfloods/lk-dev/ML',
            },
            include=['S2-pre','S2-post','ML'],
            inference_endpoint='http://127.0.0.1:8001',
            ML_models=['linear','simplecnn','unet'],
            workers=3,
            source='REST'
        )

        ingestor.ingest(ems_code=ems_code,
                        aoi_code=aoi_code,
                        code_date=event_date, 
                        aoi_geom=aoi_geom)
    
    else:
    
    
        ingestor = Ingestor(
            local_path='tmp',
            bucket='ml4floods',
            cloud_path = {
                'S2-pre':'worldfloods/lk-dev/S2-pre',
                'S2-post':'worldfloods/lk-dev/S2-post',
                'GT':'worldfloods/lk-dev/gt',
                'meta':'worldfloods/lk-dev/meta',
                'ML':'worldfloods/lk-dev/ML',
            },
            include=['S2-pre','S2-post','GT','ML'],
            inference_endpoint='http://127.0.0.1:8001',
            ML_models=['simplecnn','unet'],
            workers=3,
            source='REST'
        )

        ingestor.ingest(ems_code=ems_code,
                        aoi_code=aoi_code,
                        code_date=event_date, 
                        aoi_geom=None)

    return jsonify({'result':True})
    
    

@app.route('/get/isavailable/', methods=['GET'])
def get_isavailable():
    ems_code = request.args.get('CODE')
    AOI = request.args.get('AOI')
    client = storage.Client()
    bucket=client.bucket('ml4floods')
    fname = os.path.join('worldfloods','lk-dev','S2-post','_'.join([ems_code,AOI,'S2.tif'])) #<- need to check this
    blob = bucket.blob(fname)
    return jsonify({'exists':blob.exists()})


@app.route('/get/geometry/', methods=['GET'])
def get_geometry():
    ems_code = request.args.get('CODE')
    AOI = request.args.get('AOI')
    if 'CUSTOM' in ems_code:
        return jsonify(utils.load_json_from_bucket('ml4floods',os.path.join('worldfloods','lk-dev','meta',f'{ems_code}_{AOI}_custom.json')))
    else:
        return jsonify(utils.load_json_from_bucket('ml4floods',os.path.join('worldfloods','lk-dev','ems-aoi',f'{ems_code}_{AOI}.json')))
    

@app.route('/get/catalog/',methods=['GET'])
def refresh_catalog():
    
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    iso2 = request.args.get('iso2')
    
    start_date_object=dt.fromisoformat(start_date)
    end_date_object=dt.fromisoformat(end_date)
    
    updated_df = helpers.walk_bucket(events_df)
    gj_gdf = helpers.refresh_geojson(base_gj, updated_df)
    gj_gdf['event-date']=pd.to_datetime(gj_gdf['event-date'])
    
    # filter the df
    if iso2:
        idxs = (gj_gdf['event-date']>=start_date_object) & (gj_gdf['event-date']<end_date_object)  & gj_gdf['value'].str.contains(iso2) # change to np.prod.any in list for multi
    else:
        idxs = (gj_gdf['event-date']>=start_date_object) & (gj_gdf['event-date']<end_date_object)
        
    #updated_df['CodeDate'] = updated_df['CodeDate'].apply(lambda el: el.isoformat())
    gj_gdf['event-date'] = gj_gdf['event-date'].apply(lambda el: el.isoformat())
    
    return pd.DataFrame(gj_gdf.loc[idxs,['CODE','AOI','TITLE','event-date']]).to_json()


@app.route("/")
def hello():
    """ A <hello world> route to test server functionality. """
    return "Hello World TileServer!"

@app.route("/<layer_name>/<image_name>/<z>/<x>/<y>.png")
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
        return send_file(os.path.join(os.getcwd(),'ml4floods','serve','tileserver','static','border.png')) 
    
    try:
  
        ### get latlon bounding box from z,x,y tile request
        bounds_wgs = mercantile.bounds(int(x),int(y),int(z))  
        
        bounds_arr = [bounds_wgs.west, bounds_wgs.south, bounds_wgs.east, bounds_wgs.north]

        ################################################
        ### BY OTHERS: GET THE CLOUD PATH OF A COG TIFF
        # ... for now hardcode a single image
        #image_name = "EMSR333_02PORTOPALO_DEL_MONIT01_v1_observed_event_a"
        if 'S2' in layer_name:
            # This output shape is dictated by the tileserver. Using 256 is default.
            print ('layer name',layer_name)
            OUTPUT_SHAPE=(3,256,256)
            READBANDS=[4,3,2]
            image_address = f"gs://ml4floods/worldfloods/lk-dev/{layer_name}/{image_name}_S2.tif"
            print ('CHECK',utils.check_file_in_bucket_exists('ml4floods',f'worldfloods/lk-dev/{layer_name}/{image_name}_S2.tif'))
        elif layer_name=='mask':
            OUTPUT_SHAPE=(1,256,256)
            READBANDS=[1]
            image_address = f"gs://ml4floods/worldfloods/lk-dev/gt/{image_name}_GT.tif"
            print ('CHECK',utils.check_file_in_bucket_exists('ml4floods',f'worldfloods/lk-dev/gt/{image_name}_GT.tif'))
        elif 'ML' in layer_name:
            chosen_model = layer_name.split('_')[1]
            OUTPUT_SHAPE=(1,256,256)
            READBANDS=[1]
            image_address = f"gs://ml4floods/worldfloods/lk-dev/ML/{image_name}_ML_{chosen_model}.tif"
            print ('CHECK',utils.check_file_in_bucket_exists('ml4floods',f'worldfloods/lk-dev/ML/{image_name}_ML_{chosen_model}.tif'))
        else:
            raise

        ################################################

        # open the raster using a VRT with rasterio
        print ('image address',image_address)
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN=True, CPL_DEBUG=True):
            with rasterio.open(image_address,'r') as rst:

                print ('bounds_arr',bounds_arr)
                # get the bounds for the patch in the raster crs. (will be the same is rst.crs==epsg4326)
                rast_bounds = warp.transform_bounds({"init": "epsg:4326"}, rst.crs, *bounds_arr)
                print ('rast_bounds',rast_bounds)

                # get the pixel slice window
                window = rasterio.windows.from_bounds(*rast_bounds, rst.transform)

                print ('window',window)

                ## if out-of-range, return empty
                if (window.col_off<=-window.width) or ((window.col_off-rst.shape[0])>=window.width) or (window.row_off<=-window.height) or ((window.row_off-rst.shape[1])>=window.height) :
                    print ('out of range')
                    return send_file(os.path.join(os.getcwd(),'ml4floods','serve','tileserver','static','border.png'))  

                # use rasterio to read the pixel window from the cloud bucket
                rst_arr = rst.read(READBANDS, window=window, out_shape=OUTPUT_SHAPE, boundless=True, fill_value=0)

                print (layer_name, 'rst_arr shape',rst_arr.shape,rst_arr.min(), rst_arr.max())


        ####################################
        ### BY OTHERS: DO SOME PROCESSING, INFERENCE, ETC.
        ### Add other bands above
        
        if 'S2' in layer_name:
            ### can drive this visualisation from a SPEC in the future
            im_rgb = (np.clip(rst_arr / 2500, 0, 1).transpose((1, 2, 0)) * 255).astype(np.uint8)
        elif layer_name=='mask' or 'ML' in layer_name:
            print ('doing mask')
            # Make a blank np array of zeros
            im_rgb = np.zeros((256,256,3), dtype=np.uint8)

            # Define a color legend for the flood segmentation
            col_dict = {0:[0,0,0],1:[0,255,0],2:[0,0,255],3:[255,0,0]} # 0: nodata; 1: land; 2: water; 3: cloud

            # Loop through the color dictionary and project in the RGB legend color
            for kk,vv in col_dict.items():
                im_rgb[np.squeeze(rst_arr)==kk,:] = np.array(vv)
                
            
        # np_arr = model(np_arr, SPEC) # <- some spec document that describes the modelling pipeline, etc. so we know how to visulise it
        ####################################
        print ('im_rgb',im_rgb.shape, im_rgb.min(), im_rgb.max(), im_rgb.dtype)
        #buf = io.BytesIO()
        #Image.fromarray(im_rgb, mode="RGB").save(os.path.join(os.getcwd(),'tmp',f'{z}_{x}_{y}.png'), format="PNG") #os.path.join(os.getcwd(),'tmp',f'{z}_{x}_{y}.png')
        #Image.fromarray(im_rgb, mode="RGB").save(buf, format="JPEG")
        #buf.seek(0)
        #buf.close()
        
        buf = io.BytesIO()
        Image.fromarray(im_rgb, mode="RGB").save(buf, format="PNG") 
        buf.seek(0,0)

        return send_file(
            buf, #,os.path.join(os.getcwd(),'tmp',f'{z}_{x}_{y}.png'), #buf
            as_attachment = True,
            attachment_filename = f'{z}_{x}_{y}.png',
            mimetype = 'image/png'
        )

    except Exception as e:
        print ('ERROR!',e)
        return send_file(os.path.join(os.getcwd(),'ml4floods','serve','tileserver','static','border.png')) 
        #raise

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
