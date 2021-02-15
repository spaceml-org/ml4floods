"""
Demo script to download some demo data files. Mainly used for testing but can also be used for other explorations.
"""
import rasterio
import argparse
import subprocess
from google.cloud import storage
from pathlib import Path
from src.utils import SRC_DIR
HOME = SRC_DIR
from src.data.worldfloods.download import download_worldfloods_data




def data_download(ml_split: str="train"):

    # STEP 1 - Create Demo Directory


    # Step 2 - Download List of demo files
    bucket_id = "ml4floods"
    directory = "worldfloods/public/"

    files = [
        "01042016_Holmes_Creek_at_Vernon_FL.tif",
        "05302016_San_Jacinto_River_at_Porter_TX.tif",
        "05102017_Black_River_near_Pocahontas_AR0000012544-0000000000.tif"

        ]



    download_worldfloods_data(
        directories=files,
        destination_dir=str(Path(HOME).joinpath("datasets")),
        bucket_id=bucket_id,
        ml_split=ml_split
    )
    

if __name__ == "__main__":
    main()