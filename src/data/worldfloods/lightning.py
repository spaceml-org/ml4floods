from src.data.utils import get_files_in_directory
from typing import Tuple, Optional, List, Callable, Dict
from torch.utils.data import DataLoader
from src.data.worldfloods.dataset import WorldFloodsDatasetTiled, WorldFloodsDataset
import pytorch_lightning as pl
from pathlib import Path
from src.preprocess.tiling import WindowSize
from src.preprocess.utils import get_list_of_window_slices


class WorldFloodsDataModule(pl.LightningDataModule):
    """A prepackaged WorldFloods Pytorch-Lightning data module
    This initializes a module given a set a directory with a subdirectory
    for the training and testing data ("image_folder" and "target_folder").
    Then we can search through the directory and load the images found. It
    creates the train, val and test datasets which then can be used to initialize
    the dataloaders. This is pytorch lightning compatible which can be used with
    the training fit framework.
    
    Args:
        data_dir: (str): the top level directory where the input
            and target folder directories are found
        input_folder (str): the input folder sub_directory
        target_folder (str): the target folder sub directory
        train_transformations (Callable): the transformations used within the 
            training data module
        test_transformations (Callable): the transformations used within the
            testing data module
        window_size (Tuple[int,int]): the window size used to tile the images
            for training
        batch_size (int): the batchsize used for the dataloader
        bands (List(int)): the bands to be selected from the images
        
    Attributes:
        data_dir: (str): the top level directory where the input
            and target folder directories are found
        train_transform (Callable): the transformations used within the 
            training data module
        test_transform (Callable): the transformations used within the
            testing data module
        bands (List(int)): the bands to be selected from the images
        input_prefix (str): the input folder sub_directory
        target_prefix (str): the target folder sub directory
        window_size (Tuple[int,int]): the window size used to tile the images
                    for training
    Example:
        >>> from src.data.worldfloods.lightning import WorldFloodsGCPDataModule
        >>> wf_dm = WorldFloodsDataModule()
        >>> wf_dm.prepare_data()
        >>> wf_dm.setup()
        >>> train_dl = wf_dm.train_dataloader()
    """
    def __init__(
        self,
        data_dir: str = "./",
        input_folder: str = "S2",
        target_folder: str = "gt",
        train_transformations: Optional[Callable] = None,
        test_transformations: Optional[Callable] = None,
        window_size: Tuple[int, int] = [64, 64],
        batch_size: int = 32,
        bands: List[int] = [1, 2, 3],
        num_workers:int = 4
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_transform = train_transformations
        self.test_transform = test_transformations
        self.num_workers = num_workers
        self.num_workers_test = 0

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.bands = bands
        self.batch_size = batch_size
        # Prefixes
        self.image_prefix = input_folder
        self.gt_prefix = target_folder
        self.window_size = WindowSize(height=window_size[0], width=window_size[1])

    def prepare_data(self):
        """Does Nothing for now. Here for compatibility."""
        # TODO: here we can check for correspondence between the files
        pass

    def setup(self):
        """This creates the PyTorch dataset given the preconfigured
        file paths.
        """
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
        # TODO cache the list of window_slices
        self.train_dataset = WorldFloodsDatasetTiled(
            list_of_windows=get_list_of_window_slices(self.train_files, window_size=self.window_size),
            image_prefix=self.image_prefix,
            gt_prefix=self.gt_prefix,
            bands=self.bands,
            transforms=self.train_transform,
        )
        self.val_dataset = WorldFloodsDatasetTiled(
            list_of_windows=get_list_of_window_slices(self.val_files, window_size=self.window_size),
            image_prefix=self.image_prefix,
            gt_prefix=self.gt_prefix,
            bands=self.bands,
            transforms=self.test_transform, 
        )
        self.test_dataset = WorldFloodsDataset(
            image_files=self.test_files,
            image_prefix=self.image_prefix,
            gt_prefix=self.gt_prefix,
            bands=self.bands,
            transforms=self.test_transform,
        )

    def train_dataloader(self):
        """Initializes and returns the training dataloader"""
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        """Initializes and returns the validation dataloader"""
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=1, shuffle=False)

    def test_dataloader(self, num_workers=1):
        """Initializes and returns the test dataloader"""
        return DataLoader(self.test_dataset, batch_size=1,
                          num_workers=self.num_workers_test, shuffle=False)


class WorldFloodsGCPDataModule(pl.LightningDataModule):
    """A prepackaged WorldFloods Pytorch-Lightning data module
    This initializes a module given a GCP bucket. We define a bucket and a
    top level directory followed by subdirectories for the training and 
    testing data ("image_folder" and "target_folder").
    Then we can search through the directory and load the images found. It
    creates the train, val and test datasets which then can be used to initialize
    the dataloaders. This is pytorch lightning compatible which can be used with
    the training fit framework.
    
    Args:
        bucket_id (str): the GCP bucket name
        filenames_train_test: (str): Dict with the train, test and validation images
        input_folder (str): the input folder sub_directory
        target_folder (str): the target folder sub directory
        train_transformations (Callable): the transformations used within the 
            training data module
        test_transformations (Callable): the transformations used within the
            testing data module
        window_size (Tuple[int,int]): the window size used to tile the images
            for training
        batch_size (int): the batchsize used for the dataloader
        bands (List(int)): the bands to be selected from the images
        
    Attributes:
        data_dir: (str): the top level directory where the input
            and target folder directories are found
        train_transform (Callable): the transformations used within the 
            training data module
        test_transform (Callable): the transformations used within the
            testing data module
        bands (List(int)): the bands to be selected from the images
        input_prefix (str): the input folder sub_directory
        target_prefix (str): the target folder sub directory
        window_size (Tuple[int,int]): the window size used to tile the images
                    for training
                    
    Example:
        >>> from src.data.worldfloods.lightning import WorldFloodsGCPDataModule
        >>> wf_dm = WorldFloodsGCPDataModule()
        >>> wf_dm.prepare_data()
        >>> wf_dm.setup()
        >>> train_dl = wf_dm.train_dataloader()
    """
    def __init__(
        self,
        bucket_id: str = "ml4floods",
        filenames_train_test: Dict[str, Dict[str, List[str]]] = None,
        input_folder: str = "S2",
        target_folder: str = "gt",
        train_transformations: Optional[Callable] = None,
        test_transformations: Optional[Callable] = None,
        window_size: Tuple[int, int] = [64, 64],
        batch_size: int = 32,
        bands: List[int] = [1, 2, 3],
        num_workers: int = 4
    ):
        super().__init__()
        self.train_transform = train_transformations
        self.test_transform = test_transformations
        self.num_workers = num_workers

        # WORLDFLOODS Directories
        self.bucket_name = bucket_id
        self.train_files = filenames_train_test["train"][input_folder]
        self.val_files = filenames_train_test["val"][input_folder]
        self.test_files = filenames_train_test["test"][input_folder]

        # TODO: make this cleaner...this feels hacky.
        self.train_files = [f"gs://{self.bucket_name}/{x}" for x in self.train_files]
        self.val_files = [f"gs://{self.bucket_name}/{x}" for x in self.val_files]
        self.test_files = [f"gs://{self.bucket_name}/{x}" for x in self.test_files]

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.bands = bands
        self.batch_size = batch_size
        # Prefixes
        self.image_prefix = input_folder
        self.gt_prefix = target_folder
        self.window_size = WindowSize(height=window_size[0], width=window_size[1])

    def prepare_data(self):
        """Does Nothing for now. Here for compatibility."""
        # TODO: here we can check for correspondence between the files
        pass

    def setup(self):
        """This creates the PyTorch dataset given the preconfigured
        file paths in the bucket. This also does the tiling operations.
        """
        # get filenames from the bucket

        # add gcp dir to each of the strings

        # create datasets
        self.train_dataset = WorldFloodsDatasetTiled(
            list_of_windows=get_list_of_window_slices(self.train_files, window_size=self.window_size),
            image_prefix=self.image_prefix,
            gt_prefix=self.gt_prefix,
            transforms=self.train_transform,
            bands=self.bands,
            lock_read=True
        )
        self.val_dataset = WorldFloodsDatasetTiled(
            list_of_windows=get_list_of_window_slices(self.val_files, window_size=self.window_size),
            image_prefix=self.image_prefix,
            gt_prefix=self.gt_prefix,
            transforms=self.test_transform,
            bands=self.bands,
            lock_read=True
        )
        self.test_dataset = WorldFloodsDataset(
            image_files=self.test_files,
            image_prefix=self.image_prefix,
            gt_prefix=self.gt_prefix,
            transforms=self.test_transform,
            bands=self.bands,
            lock_read=True
        )

    def train_dataloader(self):
        """Initializes and returns the training dataloader"""
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        """Initializes and returns the validation dataloader"""
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        """Initializes and returns the test dataloader"""
        return DataLoader(self.test_dataset, batch_size=1,
                          num_workers=self.num_workers, shuffle=False)
