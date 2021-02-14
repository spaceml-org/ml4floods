import pickle

from src.data.index.map_data import MapData

class MapLayerFactory:
    def __init__(self, index_path):
        with open(index_path, 'rb') as f:
            self.index_list = pickle.load(f)

    def get_data(min_lat, min_lon, max_lat, max_lon):
        return MapData(min_lat, min_lon, max_lat, max_lon, self.index_list)
