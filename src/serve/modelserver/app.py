import io, os, logging
import json
import torch
from PIL import Image
from flask import Flask, jsonify, request, send_file, Response
import numpy as np
#import zlib
import rasterio
from rasterio.windows import Window
logging.basicConfig(level=logging.INFO)


from src.data import utils
from src.models.old import model_setup
from src.models.utils.configuration import AttrDict

app = Flask(__name__)

### load models into memory
channel_configuration_name = 'all'
inference_funcs = {}

for model_name in ['linear','unet','simplecnn']:
    opt = {
        'model': model_name,
        'device': 'cpu',
        'model_folder': os.path.join(os.getcwd(),'src','models','checkpoints',f'{model_name}'), # TODO different channel configuration means different model
        'max_tile_size': 256,
        'num_class': 3,
        'channel_configuration' : channel_configuration_name,
        'num_channels': len(model_setup.CHANNELS_CONFIGURATIONS[channel_configuration_name]),
    }
    opt = AttrDict.from_nested_dicts(opt)
    inference_funcs[model_name] = model_setup.model_inference_fun(opt)
    
N_CHANNELS = len(model_setup.CHANNELS_CONFIGURATIONS[opt.channel_configuration])

logger = logging.getLogger('ModelServer')

def get_prediction(image_bytes,model):
    arr = np.frombuffer(image_bytes, dtype=np.float32).reshape(N_CHANNELS,256,256)
    tensor = torch.Tensor(arr).unsqueeze(0)
    Y_h = inference_funcs[model](tensor)
    return torch.argmax(Y_h,dim=1).squeeze().numpy().astype(np.uint8)


@app.route("/")
def hello():
    """ A <hello world> route to test server functionality. """
    return jsonify({'status':'success','message':'Hello World!'})

@app.route('/get/tif_inference', methods=['GET'])
def tif_inference():
    """ This method opens a cloud raster and runs inference on it."""
    tif_source = request.args.get('tif_source')
    tif_dest = request.args.get('tif_dest')
    chosen_model = request.args.get('chosen_model')
    
    CODE = os.path.split(tif_source)[1].split('_')[0]
    AOI = os.path.split(tif_source)[1].split('_')[1]
    
    fname = f'{CODE}_{AOI}_ML_{chosen_model}.tif'
    
    PATCH_SIZE=512
    
    logger.info('CHECK BUCKET: '+str(utils.check_file_in_bucket_exists('ml4floods',os.path.join('worldfloods','lk-dev','S2-post',f'{CODE}_{AOI}_S2.tif'))))
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN=True, CPL_DEBUG=True):
        with rasterio.open(tif_source,'r') as rst:
            crs = rst.crs
            transform=rst.transform

            print (rst.shape)
            rgb_arr = np.zeros((3,rst.shape[0], rst.shape[1]))
            Y_arr = np.zeros((rst.shape[0], rst.shape[1]), dtype=np.uint8)
            for ii_x in range(rst.shape[1]//PATCH_SIZE +1):
                for ii_y in range(rst.shape[0]//PATCH_SIZE +1):

                    w = rst.read(list(range(1,14)), window=Window(ii_x*PATCH_SIZE, ii_y*PATCH_SIZE, 512, 512))

                    tensor = torch.Tensor(w).unsqueeze(0)
                    Y_h = inference_funcs[chosen_model](tensor)
                    Y_h = torch.argmax(Y_h,dim=1).squeeze().numpy().astype(np.uint8)

                    Y_arr[ii_y*PATCH_SIZE:min((ii_y+1)*PATCH_SIZE,rst.shape[0]),ii_x*PATCH_SIZE:min((ii_x+1)*PATCH_SIZE,rst.shape[1])] = Y_h[:min(PATCH_SIZE,rst.shape[0]-ii_y*PATCH_SIZE),:min(PATCH_SIZE,rst.shape[1]-ii_x*PATCH_SIZE)]


    logger.info(f'Done inference - {fname}')
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN=True, CPL_DEBUG=True):
        with rasterio.open(
                    os.path.join(os.getcwd(),'tmp',fname), 
                    'w', 
                    driver='COG', 
                    width=Y_arr.shape[1], 
                    height=Y_arr.shape[0], 
                    count=1,
                    dtype=Y_arr.dtype, 
                    crs=crs, 
                    transform=transform) as dst:

            dst.write(Y_arr, indexes=1)
        
    utils.save_file_to_bucket(tif_dest, os.path.join(os.getcwd(),'tmp',fname))
    
    return jsonify({'result':'success'})
    


@app.route('/<model>/predict', methods=['POST'])
def predict(model):
    print ('predict',model)
    if request.method == 'POST':
        data = request.data
        Y_h = get_prediction(image_bytes=data,model=model)
        
        ### this works!
        #buf = io.BytesIO()
        #np.save(buf, Y_h)
        #buf.seek(0)
        
        ### let's try encrypting
        buf = io.BytesIO()
        np.savez_compressed(buf, Y_h=Y_h)
        buf.seek(0)
        
    
        return send_file(
                buf,
                as_attachment = True,
                attachment_filename = f'arr.npz',
                mimetype = 'application/octet_stream'
            )


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)