import logging
import pickle
from dataclasses import dataclass
from typing import List

LOGGER = logging.getLogger(__name__)


class GeographicIndex:

    @staticmethod
    def _initialize_index() -> List[str]:
        """
        This function initializes a geographical index.
        
        We create a list of lists of strings. Each string corresponds
        to a piece of data. The list is indexed by a latitude - 90 and
        longitude - 180 using the formula (lat-90)*360 + (lon-180). 
        Transforming the 2D array to 1D using lat long.

        Args:
          None

        Returns:
          A list of lists of strings.
        """

        index_list = []
        for i in range(180 * 360):
            index_list.append([])
        return index_list

    def append_at_coords(self, lat: int, lon: int, records: list):
        """
        Append a list of records to the geographical index for a specific integral lat/lon location
        """
        flattened_idx = (lat + 90) * 360 + lon + 180
        LOGGER.debug(f"Append records to index: lat = {lat} lon = {lon} flat = {flattened_idx}")
        for rec in records:
            self.index_list[flattened_idx].append(rec)

    def save_index(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.index_list, f)

    def load_index(self, path: str):
        with open(path, 'rb') as f:
            self.index_list = pickle.load(f)

    def __init__(self):
        self.index_list = GeographicIndex._initialize_index()

