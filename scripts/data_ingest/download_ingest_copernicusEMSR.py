from ml4floods.data.copernicusEMS import activations
from ml4floods.data import utils
import os
from glob import glob
import warnings
import traceback
import sys

PATH_TO_WRITE_ZIP = 'gs://ml4cc_data_lake/0_DEV/0_Raw/WorldFloods/copernicus_ems/copernicus_ems_zip'
PATH_TO_WRITE_UNZIP = 'gs://ml4cc_data_lake/0_DEV/0_Raw/WorldFloods/copernicus_ems/copernicus_ems_unzip'
PATH_TO_WRITE_PROCESSED_DATA = "gs://ml4cc_data_lake/0_DEV/1_Staging/WorldFloods"

def main():

    import argparse

    parser = argparse.ArgumentParser('Download Copernicus EMS')
    parser.add_argument('--event_start_date', default="2015-07-01",
                        help="Event start date to consider when fetching the images from CEMS")
    parser.add_argument('--output_temp_folder', default="CopernicusEMS",
                        help="Floodmaps will be copied to bucket but downloaded also locally")
    parser.add_argument('--cems_code', default="",
                        help="CEMS Code to download images from. If empty string (default) download the images"
                             "from all the codes")
    parser.add_argument('--aoi_code', default="",
                        help="CEMS AoI to download images from. If empty string (default) download the images"
                             "from all the AoIs")
    args = parser.parse_args()
    
    # fetch Copernicus EMSR codes from Copernicus EMS activations page
    # pandas DataFrame of activations table
    table_activations_ems = activations.table_floods_ems(event_start_date=args.event_start_date)

    aoi_code_args = args.aoi_code

    if args.cems_code == "":
        emsr_codes = table_activations_ems.index.to_list()
    else:
        emsr_codes = [args.cems_code]
        assert args.cems_code in table_activations_ems.index, \
            f"CEMS code: {args.cems_code} not in table of activations\n {table_activations_ems}"

    # Folders to store the downloaded stuff
    output_temp_folder = args.output_temp_folder.replace("\\", "/")
    output_temp_folder = output_temp_folder[:-1] if output_temp_folder.endswith("/") else output_temp_folder
    output_temp_folder_unzip = f"{output_temp_folder}_raw"


    # retrieve zipfile urls for each activation EMSR code
    print(f"Generating Copernicus EMSR codes to fetch:")

    fs_bucket = utils.get_filesystem(PATH_TO_WRITE_ZIP)

    for _i, emsr_code in enumerate(emsr_codes):
        print(f"{_i+1}/{len(emsr_codes)} Processing code {emsr_code}")
        zip_files_activation_url_list = activations.fetch_zip_file_urls(emsr_code)

        code_date = table_activations_ems.loc[emsr_code]["CodeDate"]

        for _j, zip_url in enumerate(zip_files_activation_url_list):
            aoi_code = zip_url.split('_')[1]
            if aoi_code_args:
                if aoi_code != aoi_code_args:
                    continue

            name_zip = os.path.basename(zip_url)
            product_name = os.path.splitext(name_zip)[0]
            print(f"\t{_j + 1}/{len(zip_files_activation_url_list)} Processing Code {emsr_code} AoI {aoi_code} file {name_zip}")

            path_to_write_unzip_bucket = f"{PATH_TO_WRITE_UNZIP}/{emsr_code}/{aoi_code}/{product_name}/"
            path_to_write_zip_bucket = f"{PATH_TO_WRITE_ZIP}/{emsr_code}/{aoi_code}/{name_zip}"

            gcp_metadata_floodmap_dir = os.path.join(PATH_TO_WRITE_PROCESSED_DATA,
                                                      emsr_code,
                                                      aoi_code,
                                                      "flood_meta")

            gcp_floodmap_dir = os.path.join(PATH_TO_WRITE_PROCESSED_DATA,
                                            emsr_code,
                                            aoi_code,
                                            "floodmap")

            filesunzip = [f"gs://{g}" for g in fs_bucket.glob(f"{path_to_write_unzip_bucket}/*.*")]

            # Download if needed
            if not fs_bucket.exists(path_to_write_zip_bucket) or (len(filesunzip) == 0):
                zipfullpath = activations.download_vector_cems(zip_url, output_temp_folder)
                fs_bucket.put_file(zipfullpath, path_to_write_zip_bucket)
                unzipfullpath = activations.unzip_copernicus_ems(zipfullpath, output_temp_folder_unzip)

                for fextracted in glob(os.path.join(unzipfullpath, "*.*")):
                    fs_bucket.put_file(fextracted,
                                       os.path.join(path_to_write_unzip_bucket, os.path.basename(fextracted)))
            else:
                print(f"\tFile {path_to_write_zip_bucket} and {path_to_write_unzip_bucket} exists. Obtaining from bucket")
                unzipfullpath = os.path.join(output_temp_folder_unzip, product_name)
                os.makedirs(unzipfullpath, exist_ok=True)
                for fextracted in filesunzip:
                    filename_local = os.path.join(unzipfullpath, os.path.basename(fextracted))
                    if not os.path.exists(filename_local):
                        fs_bucket.get_file(fextracted, filename_local)

            # Process downloaded data
            try:
                metadata_floodmap = activations.filter_register_copernicusems(unzipfullpath,
                                                                              code_date, verbose=True)
                if metadata_floodmap is None:
                    continue

                if metadata_floodmap['ems_code'] != emsr_code:
                    print(f"\tUnexpected EMSR code { metadata_floodmap['ems_code']} expected {emsr_code}")
                    continue
                if metadata_floodmap['aoi_code'] != aoi_code:
                    print(f"\tUnexpected aoi code {metadata_floodmap['aoi_code']} expected {aoi_code}")
                    continue

                satellite_date = metadata_floodmap["satellite date"]
                gcp_metadata_floodmap_path = os.path.join(gcp_metadata_floodmap_dir,
                                                          satellite_date.strftime("%Y-%m-%d") + ".pickle")

                gcp_floodmap_path = os.path.join(gcp_floodmap_dir,
                                                 satellite_date.strftime("%Y-%m-%d") + ".geojson")

                if fs_bucket.exists(gcp_metadata_floodmap_path) and fs_bucket.exists(gcp_floodmap_path):
                    print(
                        f"\tFile {gcp_metadata_floodmap_path} and {gcp_floodmap_path} exists. will not recompute")
                    continue

                floodmap = activations.generate_floodmap(metadata_floodmap, unzipfullpath)

                utils.write_pickle_to_gcp(gs_path=gcp_metadata_floodmap_path,
                                          dict_val=metadata_floodmap)

                # push floodmap to bucket
                utils.write_geojson_to_gcp(gs_path=gcp_floodmap_path, geojson_val=floodmap)

            except Exception:
                warnings.warn(f"Failed EMSR Code: {emsr_code}  AoI Code: {aoi_code}")
                traceback.print_exc(file=sys.stdout)
    
    
if __name__ == "__main__":
    main()
