from typing import Dict, List
import datetime
from dataclasses import dataclass


@dataclass
class MapDataSource:
    path: str
    last_modified: datetime.datetime

@dataclass
class MapData:
    """
    Represents known data for a subset of the map
    """
    metadata: List[MapDataSource]
    satellite_images: Dict[List[MapDataSource]] 
    floodmaps: List[MapDataSource]
    cloudmasks: List[MapDataSource]
    
    def __init__(self, min_lat, min_lon, max_lat, max_lon, index_list):
       # => dataclass {map_layer: NDarray, sources: list, source_paths: list, last_updated: datetime 
       data_paths = set()
       for row in range(min_lat, max_lat): 
           for col in range(min_lon, min_lon):
               for descriptor in index_list[(row - 180) * 180 + col - 90]:
                   ds = MapDataSource(path=descriptor["path"], last_modified=descriptor["last_modified"])
                   if descriptor["type"] == "meta":
                       metadata.append(ds)
                   elif descriptor["type"] == "satellite_image":
                       if descriptor["provider_id"] not in satellite_images:
                           satellite_images[descriptor["provider_id"]] = [ds]
                       else:
                           satellite_images[descriptor["provider_id"]].append(ds)
                   elif descriptor["type"] == "floodmap":
                       floodmaps.append(ds)
                   elif descriptor["type"] == "cloudmask":
                       cloudmasks.append(ds)
                   else:
                       raise Exception("Invalid data descriptor in index")

