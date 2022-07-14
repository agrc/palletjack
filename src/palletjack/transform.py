"""transform.py: Processes in the Transfomation step of ETL
"""
import logging

import pandas as pd
from arcgis import GeoAccessor, GeoSeriesAccessor

from . import utils

module_logger = logging.getLogger(__name__)


class APIGeocoder:
    """Geocode using the UGRC Web API Geocoder.

    Instantiate an APIGeocoder object with an api key from developer.mapserv.utah.gov
    """

    def __init__(self, api_key):
        self.api_key = api_key
        self._class_logger = logging.getLogger(__name__).getChild(self.__class__.__name__)

    def geocode_dataframe(self, dataframe, street_col, zone_col, wkid, rate_limits=(0.015, 0.03), **api_args):
        """Geocode a pandas dataframe into a spatially-enabled dataframe

        Addresses that don't meet the threshold for geocoding (score > 70) are returned as points at 0,0

        Args:
            dataframe (pd.DataFrame): Input data with separate columns for street address and zip or city
            street_col (str): The column containing the street address
            zone_col (str): The column containing either the zip code or the city name
            wkid (int): The projection to return the x/y points in
            rate_limits(Tuple <float>): A lower and upper bound in seconds for pausing between API calls. Defaults to
            (0.015, 0.03)
            **api_args (dict): Keyword arguments to be passed as parameters in the API GET call.

        Returns:
            pd.DataFrame.spatial: Geocoded data as a spatially-enabled DataFrame
        """
        new_rows = []
        for row in dataframe.itertuples(index=False):
            row_dict = row._asdict()
            results = utils.geocode_addr(
                row_dict[street_col],
                row_dict[zone_col],
                self.api_key,
                rate_limits,
                spatialReference=str(wkid),
                **api_args
            )
            row_dict['x'], row_dict['y'], row_dict['score'], row_dict['matchAddress'] = results
            new_rows.append(row_dict)

        spatial_dataframe = pd.DataFrame.spatial.from_xy(pd.DataFrame(new_rows), 'x', 'y', sr=int(wkid))
        return spatial_dataframe
