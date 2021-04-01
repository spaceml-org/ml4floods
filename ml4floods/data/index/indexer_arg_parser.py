from argparse import ArgumentParser


class IndexerArgParser(ArgumentParser):
    """Parses Indexer command-line options and arguments."""

    def __init__(self):
        super().__init__(prog="indexer.py", description="Index pipeline output by geography and output a .pkl file for use with MapDataFactory")

        self.add_argument(
            "--worldfloods-path",
            type=str,
            default="gs://ml4floods/",
            help="Path to worldfloods directory, required only to contain tiffimages subdirectory.",
        )
        self.add_argument(
            "--output-path",
            type=str,
            help='Path of output .pkl file.',
        )
        self.add_argument(
            "--log-level",
            type=str,
            default="INFO", 
            help="One of DEBUG, INFO, WARNING, or CRITICAL"
        )
        
