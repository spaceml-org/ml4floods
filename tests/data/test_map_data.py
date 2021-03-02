import unittest
import os
from dateutil import parser

from src.data.index.geographic_index import GeographicIndex
from src.data.index.map_data_factory import MapDataFactory

class TestMapData(unittest.TestCase):

    def test_load_empty(self):
        """
        Save an empty index, load it, and ensure we are able to create
        a MapData region and that it does not contain anything.
        """
        idx = GeographicIndex()
        idx.save_index("foo.pkl")
        fac = MapDataFactory("foo.pkl")
        mdata = fac.create_map_data(12, 179, 13, 180)
        assert len(mdata.metadata) == 0
        os.remove("foo.pkl")

    @staticmethod
    def get_dummy_record_list():
        provider_geotiff_prefixes = ["S2", "L8", "gt"]
        cloudmask_prefix = "cloudprob"
        floodmap_prefix = "floodmaps"
        metadata_prefix = "meta"
        master_prefix = "worldfloods/tiffimages"
        satdate = parser.parse("2018-03-18T03:05:00Z")
        records = []
        
        records.append({"type": "meta", "last_modified": satdate, "path": os.path.join(master_prefix, metadata_prefix, "EMSR274_01AMBILOBE_DEL_v2_observed_event_a.json")})
        for provider in provider_geotiff_prefixes:
            records.append({"type": "satellite_image", "last_modified": satdate, "provider_id": provider, "path": os.path.join(master_prefix, provider, "EMSR274_01AMBILOBE_DEL_v2_observed_event_a.tif")})
        records.append({"type": "floodmap", "last_modified": satdate, "path": os.path.join(master_prefix, floodmap_prefix, "EMSR274_01AMBILOBE_DEL_v2_observed_event_a.shp")})
        records.append({"type": "cloudmask", "last_modified": satdate, "path": os.path.join(master_prefix, cloudmask_prefix, "EMSR274_01AMBILOBE_DEL_v2_observed_event_a.tif")})
        return records

    def test_load_coords_single(self):
        """
        Save an index where only a single coordinate has data and ensure
        that we are able to load this data by creating a MapData region
        including those coordinates.
        """
        idx = GeographicIndex()
        records = TestMapData.get_dummy_record_list()
        idx.append_at_coords(45, 179, records)
        idx.save_index("foo.pkl")
        fac = MapDataFactory("foo.pkl")
        mdata = fac.create_map_data(45, 179, 46, 180)
        assert len(mdata.metadata) == 1
        os.remove("foo.pkl")

    def test_load_coords_all_but_one(self):
        """
        Save an index where all except a single coordinate is filled
        with data and test that we do not see any data when we create
        a MapData region containing only that single coordinate.
        """
        idx = GeographicIndex()
        records = TestMapData.get_dummy_record_list()
        for lat in range(-90, 90):
            for lon in range(-180, 180):
                if lat != 45 and lon != 179:
                    idx.append_at_coords(lat, lon, records)
        idx.save_index("foo.pkl")
        fac = MapDataFactory("foo.pkl")
        mdata = fac.create_map_data(45, 179, 46, 180)
        assert len(mdata.metadata) == 0
        os.remove("foo.pkl")

