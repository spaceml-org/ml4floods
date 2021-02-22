import pickle
from dataclasses import dataclass
from typing import List


class GeographicIndex:

    @staticmethod
    def _initialize_index() -> List[str]:
        """
        This function initializes a geographical index.
        FIXME
        We create a list of lists of strings. Each string corresponds
        to a piece of data. The list is indexed by a latitude - 90 and
        longitude - 180 (lat-90)*360 + (lon-180). 
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

    def save_index(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.index_list, f)

    def load_index(self, path):
        with open(path, 'rb') as f:
            self.index_list = pickle.load(f)

    def __init__(self):
        self.index_list = GeographicIndex._initialize_index()

