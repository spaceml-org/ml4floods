import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import rasterio
import rasterio.windows
from torch.utils.data import Dataset

from ml4floods.data.utils import check_path_exists
from ml4floods.data.worldfloods.configs import CHANNELS_CONFIGURATIONS


def check_directory_correspondence(
	filepaths: str, image_path: str, gt_path: str
) -> None:
	# Check 1 - check the root directory exists
	# Check 2 - check the subdirectory with images exists
	# Check 3 - check the subdirectory with gt exists
	# Check 4 - Check correspondence between image and gt
	#
	"""
	if not gt_path.isdir():
	print("Mask GT does not exists for file %s" % dataset_dir)
	"""
	# Loop through filenames
	# STEP 1 - ALL CHECKS (e.g. Pathlib)
	# Check path exists
	check_path_exists(filepaths)
	# check gt path exists
	check_path_exists(Path(filepaths).joinpath(gt_path))
	# check image path exists
	check_path_exists(Path(filepaths).joinpath(image_path))
	# check correspondence between image and gt

	return None


def check_file_correspondence(
	filepath: str, image_path: str, gt_path: str, image_suffix: str
) -> None:
	# Check 1 - check the root directory exists
	# Check 2 - check the subdirectory with images exists
	# Check 3 - check the subdirectory with gt exists
	# Check 4 - Check correspondence between image and gt
	#
	"""
	if not gt_path.isdir():
	print("Mask GT does not exists for file %s" % dataset_dir)
	"""
	# Loop through filenames
	# STEP 1 - ALL CHECKS (e.g. Pathlib)
	# Check path exists
	image_files = [
		str(i.name)
		for i in Path(filepath).joinpath(image_path).glob("*." + image_suffix)
	]
	gt_files = [
		str(i.name) for i in Path(filepath).joinpath(gt_path).glob("*." + image_suffix)
	]
	msg = "Image Files are not equivalent to GT files."

	assert sorted(image_files) == sorted(gt_files), msg

	return image_files


# def is_image_raster(files: List[str], directory: str) -> (List[str], tuple):

#     bad_files = []

#     for ifile in files:
#         with rasterio.open(str(Path(directory).joinpath(ifile)), "r") as img:

#             # open file to get shape
#             shape = img.shape

#             if len(shape) != 2:
#                 bad_files.append(ifile)
#             else:
#                 continue

#     return bad_files, shape


def is_image_dim_valid(filename: str, n_dims: int) -> bool:
	
	with rasterio.open(filename, "r") as img:

		# open file to get shape
		shape = img.shape

		if len(shape) != n_dims:
			return False
		
		return True





	# bad_files = []

	# for ifile in files:
	#     with rasterio.open(str(Path(directory).joinpath(ifile)), "r") as img:

	#         # open file to get shape
	#         shape = img.shape

	#         if len(shape) != 2:
	#             bad_files.append(ifile)
	#         else:
	#             continue

	# return bad_files, shape



def is_window_size_valid(filename: str, window_size: tuple, yield_smaller_patches: bool) -> bool:
	"""
	Making sure that the window sizes (either downsampling or tiling) are smaller than the original image.

	TODO: Discard images with shape smaller than window_size

	:param      filename:      single file name to check the window size against
	:type       filename:      str
	:param      window_size:   the window size for either downsampleing or tiling
	:type       window_size:   tuple

	:returns:   { description_of_the_return_value }
	:rtype:     bool
	"""

	with rasterio.open(filename, "r") as img:
		# open file to get shape
		shape = img.shape

		if (
			(window_size is not None)
			and (not yield_smaller_patches)
			and ((shape[0] < window_size[0]) or (shape[1] < window_size[1]))
		):
			print(
				"window_size (%d, %d) too big for file %s with shape (%d, %d). Image will be discarded"
				% (window_size[0], window_size[1], dataset_dir, shape[0], shape[1])
			)
			
			return False
		
		return True	


def prepare_data_func(wf_dataset):
	"""
	Performs sanity checks and return a list of valid filenames

	:param      wf_dataset:  WorldFloodsDataset Object
	:type       wf_dataset:  Object that stores all the info about the dataset (including the dataset)
	"""

	# dataset_dict = {}
	valid_filepaths = []

	# Loop through filenames
	for dataset_dir in wf_dataset.dataset_dirs:

		# STEP 1 - check all directories exist
		check_directory_correspondence(
			dataset_dir, wf_dataset.image_prefix, wf_dataset.gt_prefix
		)

		# STEP 2 - check files are equivalent
		# this should be a list of filepaths 
		filenames = check_file_correspondence(
			dataset_dir, wf_dataset.image_prefix, wf_dataset.gt_prefix, image_suffix=wf_dataset.image_suffix
		)

		# STEP 3 - Dimension and window size checks on each image
		for file in filenames:
			file_path = str(Path(dataset_dir).joinpath(wf_dataset.image_prefix).joinpath(file))
			# check for image dimensions and see if 
			# the window sizes are smaller than the original image
			if (is_image_dim_valid(file_path, n_dims=2) and 
				is_window_size_valid(file_path, window_size=wf_dataset.window_size, yield_smaller_patches=wf_dataset.yield_smaller_patches)):
				valid_filepaths.append(file_path)

	return valid_filepaths
