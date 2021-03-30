import datetime
from dataclasses import dataclass
from typing import Dict, Set


@dataclass
class MapDataSource:
    """
    Contains information about a data source

    @field path Resource path, starting one directory above "worldfloods"
    @field last_modified Best-effort last-modified date for the asset
    """
    path: str
    last_modified: datetime.datetime

    def __hash__(self):
        return self.path.__hash__()

@dataclass
class MapData:
    """
    Represents known data for a subset of the map separated by type of data

    @field metadata Set of MapDataSources with JSON metadata
    @field satellite_images Dictionary of Sets of MapDataSources with satellite image GeoTIF data, one set per data source
    @field floodmaps Set of MapDataSources containing shape files that detail flood-related geographic information
    @field cloudmasks Set of MapDataSources with GeoTIF cloud masks
    """
    metadata: Set[MapDataSource]
    satellite_images: Dict
    floodmaps: Set[MapDataSource]
    cloudmasks: Set[MapDataSource]
    
    def __init__(self, min_lat, min_lon, max_lat, max_lon, index_list):
       self.metadata = set()
       self.satellite_images = {}
       self.floodmaps = set()
       self.cloudmasks = set()
       for row in range(min_lat, max_lat): 
           for col in range(min_lon, max_lon):
               for descriptor in index_list[(row + 90) * 360 + col + 180]:
                   ds = MapDataSource(path=descriptor["path"], last_modified=descriptor["last_modified"])
                   if descriptor["type"] == "meta":
                       self.metadata.add(ds)
                   elif descriptor["type"] == "satellite_image":
                       if descriptor["provider_id"] not in self.satellite_images:
                           self.satellite_images[descriptor["provider_id"]] = set([ds])
                       else:
                           self.satellite_images[descriptor["provider_id"]].add(ds)
                   elif descriptor["type"] == "floodmap":
                       self.floodmaps.add(ds)
                   elif descriptor["type"] == "cloudmask":
                       self.cloudmasks.add(ds)
                   else:
                       raise Exception("Invalid data descriptor in index")

