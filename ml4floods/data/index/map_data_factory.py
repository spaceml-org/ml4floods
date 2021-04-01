import pickle

from ml4floods.data.index.map_data import MapData


class MapDataFactory:
    def __init__(self, index_path: str):
        """
        Initialize a MapDataFactory from a geographic index that is capable of producing MapData objects given coordinate rectangles.
        """
        with open(index_path, 'rb') as f:
            self.index_list = pickle.load(f)

    def create_map_data(self, min_lat: int, min_lon: int, max_lat: int, max_lon: int):
        """
        Create a MapData object given a coordinate rectangle. Note that max_lat and max_lon are exclusive integral upper bounds.
        """
        return MapData(min_lat, min_lon, max_lat, max_lon, self.index_list)
