from typing import Dict
import numpy as np


@dataclass
class WorldFloodsImage:
    """Simple WFImage dataclass"""

    # ESSENTIAL METADATA
    filename: str
    bucket_name: str
    full_path: str = field(default="Not Specified")

    # PAYLOAD
    #     source_tiff: bytea = field(default = "Not Specified")
    meta_data: Dict = field(default="Not Specified")

    # BREADCRUMBS
    load_date: str = field(default=datetime.now())
    viewed_by: list = field(default_factory=list, compare=False, repr=False)
    source_system: str = field(default="Not Specified")
