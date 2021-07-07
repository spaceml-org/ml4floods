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
import sys
import os
import warnings
import traceback
import tempfile

# Geospatial modules and WorldFloods activations mapping module
import pandas as pd
from ml4floods.data.copernicusEMS import activations

from tqdm import tqdm
from ml4floods.data import utils
import fsspec
import subprocess


def main():
    # ===== Fetch ESMR Codes ==========
    fs = fsspec.filesystem("gs")
    
    csv_file = "gs://ml4cc_data_lake/0_DEV/0_Raw/WorldFloods/copernicus_ems/copernicus_ems_codes/ems_activations_20150701_20210304.csv"
    table_activations_ems = pd.read_csv(csv_file, encoding="latin1")
    table_activations_ems = table_activations_ems.set_index("Code")
    
    esmr_codes = list(table_activations_ems.index)
    
    unzipped_activations_parent_dir = "gs://ml4cc_data_lake/0_DEV/0_Raw/WorldFloods/copernicus_ems/copernicus_ems_unzip"
    data_store = "1_Staging"
    gcp_output_parent_dir = f"gs://ml4cc_data_lake/0_DEV/{data_store}/WorldFloods/"

    # ===== Generate and store registers per code ===========
    with tqdm(esmr_codes, total=len(esmr_codes), desc="Processing floodmaps") as pbar:
        for activation in pbar:
            code_date = table_activations_ems.loc[activation]["CodeDate"]
            sample_activation_dir = os.path.join(unzipped_activations_parent_dir, activation)

            aois_dirs = fs.glob(os.path.join(sample_activation_dir, "*"))

            for aoi_dir in aois_dirs:
                name_files = [os.path.basename(of).split("_observed")[0] for of in fs.glob(os.path.join(f"gs://{aoi_dir}",
                                                                                                        "*_observed*.shp"))]
                for name_file in name_files:
                    try:
                        paths_to_copy_glob = os.path.join(f"gs://{aoi_dir}", f"{name_file}*")

                        with tempfile.TemporaryDirectory(prefix=name_file) as tmpdirname:
                            subprocess.run(["gsutil", "-m", "cp", paths_to_copy_glob, tmpdirname+"/"], capture_output=True)
                            metadata_floodmap = activations.filter_register_copernicusems(tmpdirname,
                                                                                          code_date, verbose=False)
                            if metadata_floodmap is None:
                                continue

                            satellite_date = metadata_floodmap["satellite date"]
                            gcp_metadata_floodmap_path = os.path.join(gcp_output_parent_dir,
                                                                      metadata_floodmap['ems_code'],
                                                                      metadata_floodmap['aoi_code'],
                                                                      "flood_meta",
                                                                      satellite_date.strftime("%Y-%m-%d") + ".piclke")

                            gcp_floodmap_path = os.path.join(gcp_output_parent_dir,
                                                             metadata_floodmap['ems_code'],
                                                             metadata_floodmap['aoi_code'],
                                                             "floodmap",
                                                             satellite_date.strftime("%Y-%m-%d") + ".geojson")

                            if fs.exists(gcp_metadata_floodmap_path) and fs.exists(gcp_floodmap_path):
                                continue

                            floodmap = activations.generate_floodmap(metadata_floodmap, tmpdirname)

                            utils.write_pickle_to_gcp(gs_path=gcp_metadata_floodmap_path,
                                                      dict_val=metadata_floodmap)

                            # push floodmap to bucket
                            utils.write_geojson_to_gcp(gs_path=gcp_floodmap_path, geojson_val=floodmap)

                    except Exception:
                        warnings.warn(f"File {name_file} problem when computing input/output names")
                        traceback.print_exc(file=sys.stdout)

    
if __name__ == "__main__":
    main()
