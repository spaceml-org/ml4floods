import sys
from pathlib import Path

from src.utils import SRC_DIR

HOME = SRC_DIR
sys.dont_write_bytecode = True

from pprint import pprint

from src.data.worldfloods.dataset import WorldFloodsDataset
from src.data.worldfloods.test_download import data_download


def toy_data():
	print("Here!")

	
	dataset_dirs = ["src/data/demo"]
	image_prefix = "images"
	gt_prefix = "groundtruth"

	temp = WorldFloodsDataset(
		dataset_dirs=dataset_dirs, image_prefix=image_prefix, gt_prefix=gt_prefix,
		image_suffix='tif', limit=None, yield_smaller_patches=True,
	)

	print("filepaths:")
	pprint(temp.filepaths)
	print("Done with intialization!")

	print("-----------------")

	print("Tiling testing")
	print("slices:")
	pprint(temp.slices)
	print("Done!")

	print("-----------------")

	print("__getitem__ testing")
	print("getting a single item")
	temp[1]
	print("Done!")


def real_data():

	print("Downloading Sample Data...")
	ml_split = "train"
	data_download(ml_split=ml_split)
	print("Completed!")

	print("-----------------")

	print("Loading Data to Dataset...")
	dataset_dirs = [str(Path(HOME).joinpath("datasets").joinpath(ml_split))]
	image_prefix = "S2"
	gt_prefix = "gt"


	temp = WorldFloodsDataset(
		dataset_dirs=dataset_dirs, image_prefix=image_prefix, gt_prefix=gt_prefix,
		image_suffix='tif', limit=None, yield_smaller_patches=True,
	)
	print("Completed!")
	
	print("-----------------")

	print("filepaths:")
	msg = "Filepaths number is incorrect."
	assert len(temp.filepaths) == 1043, msg
	# pprint(temp.filepaths)
	print("Done with intialization!")

	print("-----------------")

	print("Tiling testing")
	print("slices:")
	msg = "Number of slices are incorrect."
	assert len(temp.filepaths) == 1043, msg
	# pprint(temp.slices)
	print("Done!")

	print("-----------------")

	print("__getitem__ testing")
	print("getting a single item")
	# print(temp[0])
	msg = "Number of slices in first element is incorrect."
	assert temp[0][0].shape == (13, 128, 128), msg
	msg = "Number of slices in first element is incorrect."
	assert temp[0][1].shape == (128, 128), msg
	print("Done!")

if __name__ == "__main__":
	real_data()
