import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import rasterio
import rasterio.windows
from torch.utils.data import Dataset

from src.data.utils import check_path_exists
from src.data.worldfloods.configs import CHANNELS_CONFIGURATIONS
from src.data.worldfloods.prepare_data import prepare_data_func


class WorldFloodsDataset(Dataset):
	"""
	A dataloader for the WorldFloods dataset.

	Attributes
	----------
	window_size: tuple(int, int)
		size of the tiling window
	image_prefix: str
		the subdirectory name for the images
	gt_prefix: str
		the subdirectory name for the groundtruth
	transform: Callable
		NOT SURE WHAT THIS IS FOR (MAYBE IMAGE TRANSFORMS)
	sample: bool
		NOT SURE WHAT THIS IS FOR
	last_filename: str
		name of the file name that was pulled last
	yield_smaller_patches: bool
		flag to indicate whether to ignore the smaller tiles 
		at the end of the image after tiling
	s2_channels: str
		the channels to choose from the original image
	dataset_dirs: str
		root directory containing all the data
	filepaths: List[str]
		the directory for the images and groundtruths
	slices: List[slice]
		list of slices to extract tiles from the original image

	"""

	def __init__(
		self,
		dataset_dirs: List[str],
		window_size: Tuple[int, int] = (128, 128),
		image_prefix: str = "/images/",
		gt_prefix: str = "/gt/",
		image_suffix: str = "tiff",
		transform: Callable = None,
		limit=None,
		sample: bool = False,
		yield_smaller_patches: bool = False,
		use_channels: List[str] = "all",
	):

		self.dataset_dirs = dataset_dirs
		self.window_size = window_size
		self.image_prefix = image_prefix
		self.gt_prefix = gt_prefix
		self.image_suffix = image_suffix
		self.transform = transform
		self.limit = limit
		self.sample = sample
		self.last_filename = None
		self.yield_smaller_patches = yield_smaller_patches

		self.s2_channels = CHANNELS_CONFIGURATIONS[use_channels]
	
		# sort to make sure that the order is deterministic 
		# (order of the flow of data points to the ML model)
		# TODO: Do this for the list of filepaths at the end as well
		self.dataset_dirs.sort()

		self.filepaths = prepare_data_func(self)
		self.slices = None

		if (self.window_size is not None) and not self.sample:
		    self._include_all(yield_smaller_patches)
		    if limit is not None:
		        vals = np.random.choice(
		            a=len(self.filepaths), size=limit, replace=False
		        )
		        self.filepaths = [self.filepaths[idx_v] for idx_v in vals]
		        self.slices = [self.slices[idx_v] for idx_v in vals]

	def _include_all(self, yield_smaller_patches=False):
		"""
		Replace the original filepaths with the tiled filenames and 
		slice indices of the tiled images.
		
		:param      yield_smaller_patches:  flag to indicate whether to ignore 
											the smaller tiles at the end of the 
											image after tiling 
		:type       yield_smaller_patches:  boolean
		"""

		tiled_filenames = []
		slices = []

		for filepath in self.filepaths:
			orig_image_shape = rasterio.open(filepath).shape
			
			# list of x,y coordinate positions for the tiles
			tile_start_coord_x = np.arange(0, orig_image_shape[0], self.window_size[0])
			tile_start_coord_y = np.arange(0, orig_image_shape[1], self.window_size[1])
			
			# tiling the entire image by storing the filepath and slices of individual tiles
			for r in tile_start_coord_x:
				for c in tile_start_coord_y:
					slice_ = (
						slice(r, min(r + self.window_size[0], orig_image_shape[0])),
						slice(c, min(c + self.window_size[1], orig_image_shape[1])),
					)

					shape_slice = tuple([s.stop - s.start for s in slice_])
					
					# ignore the tiles that are smaller than the specified window_size
					# if shape_slice != self.window_size and not yield_smaller_patches: 
					# TODO: check the logic of this negation
					if shape_slice != self.window_size and yield_smaller_patches:
						continue
					
					tiled_filenames.append(filepath)
					slices.append(slice_)

		# updating the attributes after tiling
		self.filepaths = tiled_filenames
		self.slices = slices


	def __getitem__(self, idx):
		"""
		Extract items using index values

		:param      idx:                The index
		:type       idx:                int

		:returns:   image and the ground truth
		:rtype:     tuple(ndarray, ndarray)
		"""
		x_name = self.filepaths[idx]

		self.last_filename = self.filepaths[idx]

		x_tif = rasterio.open(x_name)

		# define the window size from the tile slice
		# default window size is the same as the original image size.
		if self.window_size is None:
			window = rasterio.windows.Window.from_slices(
				slice(0, x_tif.shape[0]), slice(0, x_tif.shape[1])
			)
		else:
			slice_ = self.slices[idx]
			window = rasterio.windows.Window.from_slices(*slice_)

		# Read input as the selected channels from the original image
		channels_1_index_base_rasterio = [s + 1 for s in self.s2_channels]
		x = x_tif.read(channels_1_index_base_rasterio, window=window)

		# get rid of nan, convert to float
		x = np.nan_to_num(x).astype(np.float32)

		# Read from GT mask
		y_name = x_name.replace(self.image_prefix, self.gt_prefix, 1)
		y_tif = rasterio.open(y_name)
		if x_tif.bounds != y_tif.bounds:
			raise RuntimeError(
				f"Bounds for tif files {x_name} and {y_name} do not match"
			)

		# gt values {0: invalid, 1: land, 2: water, 3: cloud}
		y_gt = y_tif.read(window=window)

		# TODO: Need to check why the 0th index.
		y = np.nan_to_num(y_gt[0])

		# Apply transformation
		if self.transform is not None:
			res = self.transform(image=x.transpose(1, 2, 0), mask=y)
			x, y = res["image"], res["mask"]

		return x, y


	# def __len__(self):
	#     return len(self.filepaths)
