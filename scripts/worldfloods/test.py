#%%
from rasterio.plot import show

import rasterio

with rasterio.open(
    "gs://ml4cc_data_lake/0_DEV/2_Mart/worldfloods_v1_0/test/gt/EMSR286_08ITUANGONORTH_DEL_MONIT02_v1_observed_event_a.tif",
    "r",
) as f:
    print(f)

#%%
with rasterio.open(
    "gs://ml4cc_data_lake/0_DEV/2_Mart/worldfloods_v1_0/test/gt/EMSR286_08ITUANGONORTH_DEL_MONIT02_v1_observed_event_a.tif",
    "r",
) as f:
    show(f)