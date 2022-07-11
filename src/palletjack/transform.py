"""transform.py: Processes in the Transfomation step of ETL
"""

import pandas as pd
from arcgis import GeoAccessor, GeoSeriesAccessor

from . import utils


class APIGeocoder:
    """Geocode using the UGRC Web API Geocoder.

    Instantiate an APIGeocoder object with an api key from developer.mapserv.utah.gov
    """

    def __init__(self, api_key):
        self.api_key = api_key

    def geocode_dataframe(self, dataframe, street_col, zone_col, wkid):
        """Geocode a pandas dataframe into a spatially-enabled dataframe

        Addresses that don't meet the threshold for geocoding (score > 70) are returned as points at 0,0

        Args:
            dataframe (pd.DataFrame): Input data with separate columns for street address and zip or city
            street_col (str): The column containing the street address
            zone_col (str): The column containing either the zip code or the city name
            wkid (int): The projection to return the x/y points in

        Returns:
            pd.DataFrame.spatial: Geocoded data as a spatially-enabled DataFrame
        """
        dataframe[['x', 'y']] = dataframe.apply(
            utils.geocode_addr,
            axis=1,
            args=(street_col, zone_col, self.api_key),
            spatialReference=str(wkid),
            result_type='expand'
        )
        spatial_dataframe = pd.DataFrame.spatial.from_xy(dataframe, 'x', 'y', sr=int(wkid))
        return spatial_dataframe
