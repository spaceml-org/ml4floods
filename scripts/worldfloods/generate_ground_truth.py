#%%

import sys, os
from pyprojroot import here

# spyder up to find the root
root = here(project_files=[".here"])

# append to path
sys.path.append(str(here()))

import logging
import json
from src.data.create_gt import (
    generate_water_cloud_binary_gt,
    generate_land_water_cloud_gt,
)
import pandas as pd
import numpy as np
from rasterio import features
import rasterio
import geopandas as gpd
import os
from shapely.ops import cascaded_union
from src.data.utils import filter_pols, filter_land
from typing import Optional, Dict, Tuple
from src.data.config import BANDS_S2, CODES_FLOODMAP, UNOSAT_CLASS_TO_TXT

import rasterio.windows


#%%

# 1 - Demo images (str), floodmaps+S2 Image

# 1.1 - Demo Image
demo_s2_image_path = "gs://ml4floods/worldfloods/public/test/S2/EMSR286_09ITUANGOSOUTH_DEL_MONIT02_v1_observed_event_a.tif"

# 1.2  - Demo Load the stuffs
with rasterio.open(demo_s2_image_path, "r") as s2_image:
    print(s2_image)
    print(s2_image.meta)

#%%

"""TALKING"""

# 1.3 - Demo Floodmap
demo_floodmap = "gs://ml4floods/worldfloods/public/test/floodmaps/EMSR286_09ITUANGOSOUTH_DEL_MONIT02_v1_observed_event_a.shp"

# 1.4 - Load Floodmap with Geopandas
floodmap_gdf = gpd.read_file(demo_floodmap)

#%%
"""In this cell, we want to run the previous inputs through the GT script. 

We are generating the original ground truth
"""


# Run it through the GT script
gt, gt_meta = generate_land_water_cloud_gt(demo_s2_image_path, floodmap_gdf, None)
# Pray

#%%
"""In this cell, we want to run the previous inputs through the GT script. 

We are generating the new ground truth.
"""

demo_floodmap_meta = "worldfloods/public/test/meta/EMSR286_09ITUANGOSOUTH_DEL_MONIT02_v1_observed_event_a.json"

from google.cloud import storage
import json

storage_client = storage.Client()
bucket = storage_client.get_bucket("ml4floods")
blob = bucket.blob(demo_floodmap_meta)

# Download the contents of the blob as a string and then parse it using json.loads() method
floodmap_meta = json.loads(blob.download_as_string(client=None))

# Run it through the GT script
gt_binary, gt_meta_binary = generate_water_cloud_binary_gt(
    demo_s2_image_path, floodmap_gdf, floodmap_meta
)

cloud_channel, water_channel = 0, 1

# %%

from rasterio.plot import show as rasterio_plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

rasterio_plot(gt, vmin=0, vmax=3)

plt.show()


#%%
"""Clouds GT"""


fig, ax = plt.subplots()

rasterio_plot(gt_binary[cloud_channel], vmin=0, vmax=2)

plt.show()
# %%
"""Water GT"""

channel = 1
fig, ax = plt.subplots()

rasterio_plot(gt_binary[water_channel], vmin=0, vmax=2)

plt.show()

# %%
