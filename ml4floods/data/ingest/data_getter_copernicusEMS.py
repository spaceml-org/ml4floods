import logging
import os
from typing import List

import pandas
import request


def is_downloadable(url: str) -> bool:
    """
    This function checks if the url contains a downloadable resource.

    Args:
      url:
        A string containing the Copernicus Emergency Management System url.
    
    Returns:
      A boolean indicating if the url is valid.
    """

    h = requests.head(url, allow_redirects=True)
    content_type = h.headers.get('content-type').lower()
    
    return ('text' in content_type or 'html' in content_type)


def fetch_copernicusEMS_codes(ems_webpage: str, disaster_type: List[str], start_date: str):
    """
    This function takes the Copernicus Emergency Management System url
    and parses the list of activations based on disaster type and date.

    Args:
      ems_webpage (str):
        The string url for Copernicus EMS.
      disaster_type (List[str]):
        A list of strings describing the disaster type used to filter
        events from teh table of emergency activations.
      start_date (str):
        A string in 'YYYY-MM-DD' describing the starting date for the
        gathered events. The events will be filtered from that date to
        the present.

    Returns:
      ems_table:
        A dataframe containing the unique emergency code for the
        disaster event by code, title, date of code issuance, type,
        and country.
    """

    ems_table = pd.read_html(ems_webpage)[0]
    ems_table = ems_table[ems_table["Type"].isin(disaster_type)]

    ems_table = ems_table.reset_index()[['Act. Code', 'Title', 
        'Event Date', 'Type', 'Country/Terr.']]

    ems_table = ems_table.rename({"Act. Code": "Code", "Country/Terr.": \
            "Country", "Event Date": "CodeDate"}, axis=1)

    ems_table = ems_table.set_index("Code")

    return ems_table


def fetch_copernicusEMS_zipfiles():
    pass
