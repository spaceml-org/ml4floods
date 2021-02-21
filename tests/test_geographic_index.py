import unittest
import os

from src.data.index.geographic_index import GeographicIndex

class TestGeographicIndex(unittest.TestCase):

    def test_init(self):
        """
        Little more than a simple smoke test...
        Initialize the index and make sure that there is one entry in the list per degree on Earth.
        """
        idx = GeographicIndex()
        assert len(idx.index_list) == 360*180

    def test_save_load(self):
        """
        Add something simple to the index and make sure we can save/load it correctly.
        """
        idx = GeographicIndex()
        idx.append_at_coords(45, 179, ["foo", "bar"])
        idx.save_index("foobar_index.pkl")
        idx = GeographicIndex()
        idx.load_index("foobar_index.pkl")
        assert idx.index_list[(45-90)*360 + 179+180]
