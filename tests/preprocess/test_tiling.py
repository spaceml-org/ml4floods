from pathlib import Path
from typing import Optional
from src.data.utils import create_folder, get_files_in_directory
import rasterio
from src.preprocess.tiling import WindowSize, save_tiles
import sys, os
from pyprojroot import here

ROOT = here(project_files=[".here"])


def test_save_tiles(test_dir: Optional[str] = None):

    if test_dir is None:
        test_dir = "./"

    # ===========================
    # Demo Image
    # ===========================
    gs_index = "gs://"
    bucket_id = "ml4floods"
    path = "worldfloods/public/"
    sub_dir = "train/S2"
    file_name = "01042016_Holmes_Creek_at_Vernon_FL.tif"

    # ===========================
    # Parameters
    # ===========================
    file_name = "gs://ml4floods/worldfloods/public/train/S2/01042016_Holmes_Creek_at_Vernon_FL.tif"

    window_size = WindowSize(height=64, width=64)
    dest_dir = str(Path(ROOT).joinpath("datasets/test"))
    create_folder(dest_dir)
    bands = [1, 2, 3]
    verbose = True
    n_samples = 10
    print(file_name)
    print(dest_dir)

    # ===========================
    # Save Tiles
    # ===========================

    save_tiles(
        file_name=str(file_name),
        dest_dir=str(dest_dir),
        bands=bands,
        window_size=window_size,
        verbose=verbose,
        n_samples=n_samples,
    )

    # ===========================
    # Check # Tiles
    # ===========================
    assert len(get_files_in_directory(dest_dir, "tif")) == n_samples


if __name__ == "__main__":
    test_save_tiles()