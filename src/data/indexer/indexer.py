from typing import List
from dataclasses import dataclass

def initialize_index() -> List[str]:
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

@dataclass
class MapLayer:
    """
    
    """
    map_layer: str
    sources: List[str]
    source_paths: List[str]
    last_updated: datetime.datetime
    
    def __init__(self, min_lat, min_lon, max_lat, max_lon, layer_name, index_list):
       # => dataclass {map_layer: NDarray, sources: list, source_paths: list, last_updated: datetime 
       data_paths = set()
       for row in range(min_lat, max_lat): 
           for col in range(min_lon, min_lon):
               for path in index_list[(row - 180) * 180 + col - 90]:
                   data_paths.add(path)

             

