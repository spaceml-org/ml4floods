# 1 - Fetch copernicus_ems_code/*.csv file to query

# 2 - call filter_register_copernicusems to fetch .shp
    ##### INPUT
    # - unzipped/extracted directory path
    # - code date (from copernicus_ems_code)
    ##### OUT
    # - metadata_floodmap TBD geojson?????

# 4 - generate_floodmap
    # - metadata_floodmap
    # - directory location which is in metadata_floodmap
    
# Path based modules to allow us to save files to our local machine
from pyprojroot import here
import sys
import os
root = here(project_files=[".here"])
sys.path.append(str(here()))

# Geospatial modules and WorldFloods activations mapping module
import geopandas as gpd
import pickle
import pandas as pd
from src.data.copernicusEMS import activations
from src.data import utils

from pprint import pprint
from google.cloud import storage
from io import BytesIO

import json
import geopandas as gpd
import subprocess
from tqdm import tqdm
from src.data import utils

def main():
    # ===== Fetch ESMR Codes ==========
    
    csv_file = "gs://ml4cc_data_lake/0_DEV/0_Raw/WorldFloods/copernicus_ems/copernicus_ems_codes/ems_activations_20150701_20210304.csv"
    table_activations_ems = pd.read_csv(csv_file, encoding="latin1")
    table_activations_ems = table_activations_ems.set_index("Code")
    
    esmr_codes = list(table_activations_ems.index)
    
    unzipped_activations_parent_dir = "gs://ml4cc_data_lake/0_DEV/0_Raw/WorldFloods/copernicus_ems/copernicus_ems_unzip"
    data_store = "1_Staging"
    gcp_output_parent_dir = f"gs://ml4cc_data_lake/0_DEV/{data_store}/WorldFloods/"

    # ===== Generate and store registers per code ===========
    with tqdm(esmr_codes[76:]) as pbar:
        for activation in pbar:
            code_date = table_activations_ems.loc[activation]["CodeDate"]
            sample_activation_dir = os.path.join(unzipped_activations_parent_dir, activation)
            register_list = activations.filter_register_copernicusems_gcp(sample_activation_dir, code_date)

            # ====== Create and save metadata_floodmap and floodmap per AOI =======
            for metadata_floodmap in register_list:
                floodmap = activations.generate_floodmap_gcp(metadata_floodmap, folder_files=unzipped_activations_parent_dir)

                # push metadata to bucket
                gcp_metadata_floodmap_path = os.path.join(gcp_output_parent_dir,
                                                          "flood_meta",
                                                          metadata_floodmap['ems_code'], 
                                                          metadata_floodmap['aoi_code'],
                                                          f"{metadata_floodmap['event_id']}_metadata_floodmap.pickle")
                utils.write_pickle_to_gcp(gs_path=gcp_metadata_floodmap_path, dict_val=metadata_floodmap)

                # push floodmap to bucket
                gcp_floodmap_path = os.path.join(gcp_output_parent_dir,
                                                 "floodmap",
                                                 metadata_floodmap['ems_code'], 
                                                 metadata_floodmap['aoi_code'],
                                                 f"{metadata_floodmap['event_id']}_floodmap.geojson")
                utils.write_geojson_to_gcp(gs_path=gcp_floodmap_path, geojson_val=floodmap)


    
if __name__ == "__main__":
    main()