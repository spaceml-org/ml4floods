

# CONSTANTS TO DEFINE CONFIGURATION FILES (TO BE USED LATED MAYBE)

# TASK = 'segmentation' # e.g segmentation, classification, generation, inundation, etc...
# SUB_TASK = 'flood_extent' # e.g flood_extent, cloud_removal, permanent_water, etc...
# DATA_TYPE = 'multi_band_optical' # e.g multi_band_optical, single_band_optical, multi_band_radar, dem, dtm, etc...
# TARGET_TYPE = 'multi_class_segmentation' # e.g multi_class_segmentation, binary_segmentation, etc...
# PRE_PROCESSING = ['select_bands', 'normalization'] # e.g select_bands, normalization, etc...
# MODEL_ARCHITECTURE = 'unet' # e.g unet, simplecnn, linearcnn, etc...
# MODEL_CHECKPOINT = 'final' # e.g final, best, checkpoint_0, None, etc...
# METRICS = ['confusion_matrix'] # e.g confusion_matrix, dice, precision, recall, etc...

# opt = {
#     'task': TASK,
#     'sub_task': SUB_TASK,
#     'data_type': DATA_TYPE,
#     'target_type': TARGET_TYPE,
#     'pre_processing': PRE_PROCESSING,
#     'model_architecture': MODEL_ARCHITECTURE,
#     'model_checkpoint': MODEL_CHECKPOINT,
#     'metrics': METRICS,
#     'device_parameters': {
#         'device': 'cpu'
#     },
#     'model_parameters': {
#         'model_folder': f'../pipelines/models/checkpoints/{MODEL_ARCHITECTURE}/',
#         'max_tile_size': 128,
#         'num_class': 3,
#         'channel_configuration': 'all',
#         'num_channels': len(model_setup.CHANNELS_CONFIGURATIONS['all'])
#     }
# }
# opt = AttrDict.from_nested_dicts(opt)

class AttrDict(dict):
    """ Dictionary subclass whose entries can be accessed like attributes
        (as well as normally).
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def from_nested_dicts(data):
        """ Construct nested AttrDicts from nested dictionaries. """
        if not isinstance(data, dict):
            return data
        else:
            return AttrDict({key: AttrDict.from_nested_dicts(data[key])
                             for key in data})