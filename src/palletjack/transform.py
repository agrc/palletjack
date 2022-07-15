"""transform.py: Processes in the Transfomation step of ETL
"""
import logging
import warnings
from datetime import datetime

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

        start = datetime.now()

        #: Should this return? Should it raise an error instead?
        if dataframe.empty:
            warnings.warn('No records to geocode (empty dataframe)', RuntimeWarning)

        self._class_logger.debug('Renaming columns to conform to ArcGIS limitations, if necessary...')
        dataframe.rename(columns=utils.rename_columns_for_agol(dataframe.columns), inplace=True)

        dataframe_length = len(dataframe.index)
        reporting_interval = utils.calc_modulus_for_reporting_interval(dataframe_length)
        self._class_logger.info('Geocoding %s rows...', dataframe_length)

        new_rows = []
        for i, row in enumerate(dataframe.itertuples(index=False)):
            if i % reporting_interval == 0:
                self._class_logger.info('Geocoding row %s of %s, %s%%', i, dataframe_length, i / dataframe_length * 100)
            row_dict = row._asdict()
            results = utils.geocode_addr(
                row_dict[street_col],
                row_dict[zone_col],
                self.api_key,
                rate_limits,
                spatialReference=str(wkid),
                **api_args
            )
            self._class_logger.debug(
                '%s of %s: %s, %s = %s', i, dataframe_length, row_dict[street_col], row_dict[zone_col], results
            )
            row_dict['x'], row_dict['y'], row_dict['score'], row_dict['matchAddress'] = results
            new_rows.append(row_dict)

        spatial_dataframe = pd.DataFrame.spatial.from_xy(pd.DataFrame(new_rows), 'x', 'y', sr=int(wkid))

        end = datetime.now()
        self._class_logger.info('%s Records geocoded in %s', len(spatial_dataframe.index), (end - start))
        try:
            self._class_logger.debug('Average time per record: %s', (end - start) / len(spatial_dataframe.index))
        except ZeroDivisionError:
            warnings.warn('Empty spatial dataframe after geocoding', RuntimeWarning)
        return spatial_dataframe
