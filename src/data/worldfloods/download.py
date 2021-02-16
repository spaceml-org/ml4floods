from pathlib import Path
from typing import List, Optional

from src.data.utils import save_file_from_bucket
from src.utils import SRC_DIR

HOME = SRC_DIR

BUCKET_ID = "ml4floods"
DIR = "worldfloods/public/"


def download_worldfloods_data(directories: List[str], destination_dir: str, ml_split: str="train", bucket_id: Optional[str]=None) -> None:

    
    if bucket_id is None:
        bucket_id = BUCKET_ID

    for ifile in directories:
        for iprefix in ["S2", "gt"]:

            # where to grab the file
            source = str(Path(DIR).joinpath(ml_split).joinpath(iprefix).joinpath(ifile))
            # Image where to save the file
            destination = Path(destination_dir).joinpath(ml_split).joinpath(iprefix)
            # copy file from bucket to savepath
            save_file_from_bucket(bucket_id=BUCKET_ID, file_name=source, destination_file_path=str(destination))

