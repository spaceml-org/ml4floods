import io
import json

import numpy as np
import torch
from flask import Flask, Response, jsonify, request, send_file
from PIL import Image

from src.models.config_setup import get_default_config
from src.models.worldfloods_model import WorldFloodsModel
from src.models.model_setup import get_model_inference_function
from src.data.worldfloods.configs import CHANNELS_CONFIGURATIONS

app = Flask(__name__)

### load models into memory
channel_configuration_name = 'all'
inference_funcs = {}

models_conf = {
    "simplecnn": {
        "experiment_name" : "WFV1_scnn20",
        "checkpoint_name" : "epoch=5-step=24581.ckpt",
    },
    "unet": {
        "experiment_name": "WFV1_unet",
        "checkpoint_name": "epoch=24-step=153649.ckpt",
    }
}

for model_name, conf in models_conf.items():
    opt = get_default_config(f"gs://ml4cc_data_lake/0_DEV/2_Mart/2_MLModelMart/{conf['experiment_name']}/config.json")
    checkpoint_path = f"gs://ml4cc_data_lake/0_DEV/2_Mart/2_MLModelMart/{conf['experiment_name']}/checkpoint/{conf['checkpoint_name']}"
    model = WorldFloodsModel.load_from_checkpoint(checkpoint_path)

    # Set up to control the max size of the tiles to predict
    opt["model_params"]["hyperparameters"]["max_tile_size"] = 256
    inference_function = get_model_inference_function(model, opt, apply_normalization=True)

    inference_funcs[model_name] = inference_function
    N_CHANNELS = len(CHANNELS_CONFIGURATIONS[opt.channel_configuration])


def get_prediction(image_bytes,model):
    arr = np.frombuffer(image_bytes, dtype=np.float32).reshape(N_CHANNELS,256,256)
    tensor = torch.Tensor(arr).unsqueeze(0)
    Y_h = inference_funcs[model](tensor)
    return torch.argmax(Y_h,dim=1).squeeze().numpy().astype(np.uint8)


@app.route("/")
def hello():
    """ A <hello world> route to test server functionality. """
    return "Hello World ModelServer!"


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