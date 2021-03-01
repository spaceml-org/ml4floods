from src.data.utils import get_files_in_bucket_directory, get_files_in_directory
from typing import Tuple, Optional, List, Callable
from torch.utils.data import DataLoader
import albumentations
from src.data.worldfloods.dataset import WorldFloodsDatasetTiled
import pytorch_lightning as pl
from pathlib import Path


class WorldFloodsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        input_folder: str = "S2",
        target_folder: str = "gt",
        data_dir: str = "./",
        train_transformations: Optional[List[Callable]] = None,
        test_transformations: Optional[List[Callable]] = None,
        window_size: Tuple[int, int] = [64, 64],
        batch_size: int = 32,
        bands: List[int] = [1, 2, 3],
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_transform = (
            albumentations.Compose(train_transformations)
            if train_transformations is not None
            else None
        )
        self.test_transform = (
            albumentations.Compose(test_transformations)
            if test_transformations is not None
            else None
        )

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.bands = bands
        self.batch_size = batch_size
        # Prefixes
        self.image_prefix = input_folder
        self.gt_prefix = target_folder
        self.window_size = window_size

    def prepare_data(self):
        # TODO: create the train/test/val structure
        # TODO: here we can check for correspondence between the files
        pass

    def setup(self):

        # get the path names
        files = {}
        splits = ["train", "test", "val"]

        # loop through the naming splits
        for isplit in splits:

            # get the subdirectory
            sub_dir = Path(self.data_dir).joinpath(isplit).joinpath(self.image_prefix)
            # append filenames to split dictionary
            files[isplit] = get_files_in_directory(sub_dir, "tif")

        # save filenames
        self.train_files = files["train"]
        self.val_files = files["val"]
        self.test_files = files["test"]

        # create datasets
        self.train_dataset = WorldFloodsDatasetTiled(
            image_files=self.train_files,
            image_prefix=self.image_prefix,
            gt_prefix=self.gt_prefix,
            window_size=self.window_size,
            bands=self.bands,
            transforms=self.train_transform,
        )
        # TODO: Clarify whether validations set should use augmentation or not
        self.val_dataset = WorldFloodsDatasetTiled(
            image_files=self.val_files,
            image_prefix=self.image_prefix,
            gt_prefix=self.gt_prefix,
            bands=self.bands,
            window_size=self.window_size,
            transforms=self.test_transform, 
        )
        self.test_dataset = WorldFloodsDatasetTiled(
            image_files=self.test_files,
            image_prefix=self.image_prefix,
            gt_prefix=self.gt_prefix,
            bands=self.bands,
            window_size=self.window_size,
            transforms=self.test_transform,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class WorldFloodsGCPDataModule(pl.LightningDataModule):
    def __init__(
        self,
        bucket_id: str = "ml4floods",
        path_to_splits: str = "worldfloods/public",
        input_folder: str = "S2",
        target_folder: str = "gt",
        train_transformations: Optional[List[Callable]] = None,
        test_transformations: Optional[List[Callable]] = None,
        window_size: Tuple[int, int] = [64, 64],
        batch_size: int = 32,
        bands: List[int] = [1, 2, 3],
    ):
        super().__init__()
        self.train_transform = (
            albumentations.Compose(train_transformations)
            if train_transformations is not None
            else None
        )
        self.test_transform = (
            albumentations.Compose(test_transformations)
            if test_transformations is not None
            else None
        )

        # WORLDFLOODS Directories
        self.bucket_name = bucket_id
        self.train_dir = f"{path_to_splits}/train/{input_folder}"
        self.val_dir = f"{path_to_splits}/val/{input_folder}"
        self.test_dir = f"{path_to_splits}/test/{input_folder}"

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.bands = bands
        self.batch_size = batch_size
        # Prefixes
        self.image_prefix = input_folder
        self.gt_prefix = target_folder
        self.window_size = window_size

    def prepare_data(self):
        # TODO: potentially download the data
        # TODO: create the train/test/val structure
        # TODO: here we can check for correspondence between the files
        pass

    def setup(self):

        # get filenames from the bucket
        self.train_files = get_files_in_bucket_directory(
            self.bucket_name, self.train_dir, ".tif"
        )
        self.val_files = get_files_in_bucket_directory(
            self.bucket_name, self.val_dir, ".tif"
        )
        self.test_files = get_files_in_bucket_directory(
            self.bucket_name, self.test_dir, ".tif"
        )

        # add gcp dir to each of the strings
        # TODO: make this cleaner...this feels hacky.
        self.train_files = [f"gs://{self.bucket_name}/{x}" for x in self.train_files]
        self.val_files = [f"gs://{self.bucket_name}/{x}" for x in self.val_files]
        self.test_files = [f"gs://{self.bucket_name}/{x}" for x in self.test_files]

        # create datasets
        self.train_dataset = WorldFloodsDatasetTiled(
            image_files=self.train_files,
            image_prefix=self.image_prefix,
            gt_prefix=self.gt_prefix,
            window_size=self.window_size,
            transforms=self.train_transform,
            bands=self.bands
        )
        self.val_dataset = WorldFloodsDatasetTiled(
            image_files=self.val_files,
            image_prefix=self.image_prefix,
            gt_prefix=self.gt_prefix,
            window_size=self.window_size,
            transforms=self.test_transform,
            bands=self.bands
        )
        self.test_dataset = WorldFloodsDatasetTiled(
            image_files=self.test_files,
            image_prefix=self.image_prefix,
            gt_prefix=self.gt_prefix,
            window_size=self.window_size,
            transforms=self.test_transform,
            bands=self.bands
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)