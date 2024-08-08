"""Transform pandas dataframes in preparation for loading to AGOL."""

import locale
import logging
import warnings
from datetime import datetime

import arcgis
import pandas as pd
from arcgis import GeoAccessor, GeoSeriesAccessor  # noqa: F401

from palletjack import utils

module_logger = logging.getLogger(__name__)


class APIGeocoder:
    """Geocode a dataframe using the UGRC Web API Geocoder.

    Instantiate an APIGeocoder object with an api key from developer.mapserv.utah.gov. It will attempt to validate the
    API key. If validation fails, it will raise one of the following errors:

    - RuntimeError: If there was a network or other error
    - ValueError: If the key is invalid
    - UserWarning: If the API responds with some other abnormal result

    The individual geocoding steps are exposed in the `palletjack.utils.Geocoding` class in the utils module for use in
    other settings.
    """

    def __init__(self, api_key):
        """
        Args:
            api_key (str): API key obtained from developer.mapserv.utah.gov
        """
        self.api_key = api_key
        self._class_logger = logging.getLogger(__name__).getChild(self.__class__.__name__)
        utils.Geocoding.validate_api_key(self.api_key)

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
            warnings.warn("No records to geocode (empty dataframe)", RuntimeWarning)

        dataframe_length = len(dataframe.index)
        reporting_interval = utils.calc_modulus_for_reporting_interval(dataframe_length)
        self._class_logger.info("Geocoding %s rows...", dataframe_length)

        street_col_index = dataframe.columns.get_loc(street_col)
        zone_col_index = dataframe.columns.get_loc(zone_col)
        new_rows = []
        for i, row in enumerate(dataframe.itertuples(index=False)):
            if i % reporting_interval == 0:
                self._class_logger.info("Geocoding row %s of %s, %s%%", i, dataframe_length, i / dataframe_length * 100)
            row_dict = row._asdict()
            results = utils.Geocoding.geocode_addr(
                row[street_col_index],
                row[zone_col_index],
                self.api_key,
                rate_limits,
                spatialReference=str(wkid),
                **api_args,
            )
            self._class_logger.debug(
                "%s of %s: %s, %s = %s", i, dataframe_length, row[street_col_index], row[zone_col_index], results
            )
            row_dict["x"], row_dict["y"], row_dict["score"], row_dict["matchAddress"] = results
            new_rows.append(row_dict)

        spatial_dataframe = pd.DataFrame.spatial.from_xy(pd.DataFrame(new_rows), "x", "y", sr=int(wkid))

        end = datetime.now()
        self._class_logger.info("%s Records geocoded in %s", len(spatial_dataframe.index), (end - start))
        try:
            self._class_logger.debug("Average time per record: %s", (end - start) / len(spatial_dataframe.index))
        except ZeroDivisionError:
            warnings.warn("Empty spatial dataframe after geocoding", RuntimeWarning)
        return spatial_dataframe


class FeatureServiceMerging:
    """Get the live dataframe from a feature service and update it from another dataframe"""

    @staticmethod
    def update_live_data_with_new_data(live_dataframe, new_dataframe, join_column):
        """Update a dataframe with data from another

        Args:
            live_dataframe (pd.DataFrame): The dataframe containing info to be updated
            new_dataframe (pd.DataFrame): Dataframe containing source info to use in the update
            join_column (str): The column with unique IDs to be used as a key between the two dataframes

        Raises:
            ValueError: If the join_column is missing from either live or new data
            RuntimeWarning: If there are rows in the new data that are not found in the live data; these will not be
                added to the live dataframe.

        Returns:
            pd.DataFrame: The updated dataframe, with data types converted via .convert_dtypes()
        """

        try:
            live_dataframe.set_index(join_column, inplace=True)
            new_dataframe.set_index(join_column, inplace=True)
        except KeyError as error:
            raise ValueError("Join column not found in live or new dataframes") from error

        indicator_dataframe = live_dataframe.merge(new_dataframe, on=join_column, how="outer", indicator=True)
        new_only_dataframe = indicator_dataframe[indicator_dataframe["_merge"] == "right_only"]
        if not new_only_dataframe.empty:
            keys_not_found = list(new_only_dataframe.index)
            warnings.warn(
                f"The following keys from the new data were not found in the existing dataset: {keys_not_found}",
                RuntimeWarning,
            )

        live_dataframe.update(new_dataframe)
        return live_dataframe.reset_index().convert_dtypes()

    @staticmethod
    def get_live_dataframe(gis, feature_service_itemid, layer_index=0):
        """Get a spatially-enabled dataframe representation of a hosted feature layer

        Args:
            gis (arcgis.gis.GIS): GIS object of the desired organization
            feature_service_itemid (str): itemid in the gis of the desired hosted feature service
            layer_index (int, optional): Index of the desired layer within the hosted feature service. Defaults to 0.

        Raises:
            RuntimeError: If it fails to load the data

        Returns:
            pd.DataFrame.spatial: Spatially-enabled dataframe representation of the hosted feature layer
        """

        try:
            feature_layer = arcgis.features.FeatureLayer.fromitem(
                gis.content.get(feature_service_itemid), layer_id=layer_index
            )
            live_dataframe = feature_layer.query(as_df=True)
        except Exception as error:
            raise RuntimeError("Failed to load live dataframe") from error

        return live_dataframe


class DataCleaning:
    """Static methods for cleaning dataframes prior to uploading to AGOL"""

    @staticmethod
    def switch_to_nullable_int(dataframe, fields_that_should_be_ints):
        """Convert specified fields to panda's nullable Int64 type to preserve int to EsriFieldTypeInteger mapping

        Args:
            dataframe (pd.DataFrame): Input dataframe with columns to be converted
            fields_that_should_be_ints (list[str]): List of column names to be converted

        Raises:
            TypeError: If any of the conversions fail. Often caused by values that aren't int-castable floats (ie. x.0)
                or np.nans.

        Returns:
            pd.DataFrame: Input dataframe with columns converted to nullable Int64
        """

        retyped = dataframe.copy()
        try:
            for field in fields_that_should_be_ints:
                retyped[field] = DataCleaning._switch_series_to_numeric_dtype(retyped[field], "Int64")
        except (TypeError, ValueError) as error:
            raise TypeError(
                "Cannot convert one or more fields to nullable ints. Check for non-int/non-np.nan values."
            ) from error
        return retyped

    @staticmethod
    def switch_to_float(dataframe, fields_that_should_be_floats):
        """Convert specified fields to float, converting empty strings to None first as required

        Args:
            dataframe (pd.DataFrame): Input dataframe with columns to be converted
            fields_that_should_be_floats (list[str]): List of column names to be converted

        Raises:
            TypeError: If any of the conversions fail. Often caused by values that aren't castable to floats
                (non-empty, non-numeric strings, etc)

        Returns:
            pd.DataFrame: Input dataframe with columns converted to float
        """

        retyped = dataframe.copy()
        try:
            for field in fields_that_should_be_floats:
                retyped[field] = DataCleaning._switch_series_to_numeric_dtype(retyped[field], "float")
        except (TypeError, ValueError) as error:
            raise TypeError(
                "Cannot convert one or more fields to floats. Check for non-float/non-null values."
            ) from error
        return retyped

    @staticmethod
    def _switch_series_to_numeric_dtype(series, dtype):
        """Switch the dtype of a series to the specified dtype

        Series of dtype 'object' (ie, series of strings or mixed strings and numbers) are converted to str so that they
        can be de-localized to remove comma thousands separators

        Args:
            series (pd.Series): The series to be converted
            dtype (str): The dtype to convert to

        Returns:
            pd.Series: The converted series
        """

        if series.dtype == "object":
            series = series.astype(str).apply(locale.delocalize)
            series.replace("", None, inplace=True)
        return series.astype(dtype)

    @staticmethod
    def switch_to_datetime(dataframe, date_fields, **to_datetime_kwargs):
        """Convert specified fields to datetime dtypes to ensure proper date formatting for AGOL

        Args:
            dataframe (pd.DataFrame): The source dataframe
            date_fields (List[int]): The fields to convert to datetime
            **to_datetime_kwargs (keyword arguments, optional): Arguments to pass through to pd.to_datetime

        Returns:
            pd.DataFrame: The source dataframe with converted fields.
        """

        for field in date_fields:
            dataframe[field] = (
                pd.to_datetime(dataframe[field], **to_datetime_kwargs).dt.as_unit("ns").dt.tz_localize(None)
            )

        return dataframe

    @staticmethod
    def rename_dataframe_columns_for_agol(dataframe):
        """Rename all the columns in a dataframe to valid AGOL column names

        Args:
            dataframe (pd.DataFrame): Dataframe to be renamed

        Returns:
            pd.DataFrame: Input dataframe with renamed columns
        """

        rename_dict = utils.rename_columns_for_agol(dataframe.columns)
        renamed_dataframe = dataframe.rename(columns=rename_dict)
        return renamed_dataframe
