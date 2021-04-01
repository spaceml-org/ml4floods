from argparse import ArgumentParser


class UnosatDownloadArgParser(ArgumentParser):

    def __init__(self):
        super().__init__(
            prog="unosat_download.py", description="Download UNOSAT data to buckets"
        )

        self.add_argument(
            "--base-url",
            type=str,
            default="https://unitar.org",
            help="Base URL where UNITAR maps are hosted",
        )

        self.add_argument(
            "--country-list-url",
            type=str,
            default="https://unitar.org/maps/countries",
            help="URL of UNITAR country list for accessing map data",
        )

        self.add_argument(
            "--download-base-regex",
            type=str,
            default="https?://unosat-maps\.web\.cern\.ch/",
            help="RegEx for UNOSAT map download server",
        )

        self.add_argument(
            "--shapefile-bucket",
            type=str,
            default="gs://ml4cc_data_lake/0_DEV/0_Raw/WorldFloods/shapefiles/unosat/",
            help="Path in FS or a major cloud (S3, GCS, Azure Blob) to store shapfiles",
        )

        self.add_argument(
            "--metadata-bucket",
            type=str,
            default="gs://ml4cc_data_lake/0_DEV/0_Raw/WorldFloods/meta/",
            help="Path in FS or a major cloud (S3, GCS, Azure Blob) to store shapfile metadata",
        )

