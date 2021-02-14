from typing import List
from dataclasses import dataclass


@dataclass
class MapData:
    """
    
    """
    map_layer: str
    sources: List[str]
    source_paths: List[str]
    last_updated: datetime.datetime

    
    def __init__(self, min_lat, min_lon, max_lat, max_lon, index_list):
       # => dataclass {map_layer: NDarray, sources: list, source_paths: list, last_updated: datetime 
       data_paths = set()
       for row in range(min_lat, max_lat): 
           for col in range(min_lon, min_lon):
               for path in index_list[(row - 180) * 180 + col - 90]:
                   data_paths.add(path)

             

