"""Utility classes and methods that are used internally throughout palletjack. Many are exposed publicly in case they are useful elsewhere in a client's code."""

import datetime
import importlib
import logging
import random
import re
import sys
import warnings
from math import floor
from pathlib import Path
from time import sleep

import arcgis
import geopandas as gpd
import numpy as np
import pandas as pd
import pygsheets
import pyogrio
import requests

from palletjack.errors import IntFieldAsFloatError, TimezoneAwareDatetimeError

module_logger = logging.getLogger(__name__)

RETRY_MAX_TRIES = 3
RETRY_DELAY_TIME = 2


def retry(worker_method, *args, **kwargs):
    """Allows you to retry a function/method to overcome network jitters or other transient errors.

    Retries worker_method RETRY_MAX_TRIES times (for a total of n+1 tries, including the initial attempt), pausing
    2^RETRY_DELAY_TIME seconds between each retry. Any arguments for worker_method can be passed in as additional
    parameters to retry() following worker_method: retry(foo_method, arg1, arg2, keyword_arg=3).

    RETRY_MAX_TRIES and RETRY_DELAY_TIME default to 3 tries and 2 seconds, but can be overridden by setting the
    palletjack.utils.RETRY_MAX_TRIES and palletjack.utils.RETRY_DELAY_TIME constants in the client script.

    Args:
        worker_method (callable): The name of the method to be retried (minus the calling parens)

    Raises:
        error: The final error that causes worker_method to fail after 3 retries

    Returns:
        various: The value(s) returned by worked_method
    """
    tries = 1
    max_tries = RETRY_MAX_TRIES
    delay = RETRY_DELAY_TIME  #: in seconds

    #: this inner function (closure? almost-closure?) allows us to keep track of tries without passing it as an arg
    def _inner_retry(worker_method, *args, **kwargs):
        nonlocal tries

        try:
            return worker_method(*args, **kwargs)

        #: ArcGIS API for Python loves throwing bog-standard Exceptions, so we can't narrow this down further
        except Exception as error:
            if tries <= max_tries:  # pylint: disable=no-else-return
                wait_time = delay**tries
                module_logger.debug(
                    'Exception "%s" thrown on "%s". Retrying after %s seconds...', error, worker_method, wait_time
                )
                sleep(wait_time)
                tries += 1
                return _inner_retry(worker_method, *args, **kwargs)
            else:
                raise error

    return _inner_retry(worker_method, *args, **kwargs)


def rename_columns_for_agol(columns):
    """Replace special characters and spaces with '_' to match AGOL field names

    Args:
        columns (iter): The new columns to be renamed

    Returns:
        Dict: Mapping {'original name': 'cleaned_name'}
    """

    rename_dict = {}
    for column in columns:
        no_specials = re.sub(r"[^a-zA-Z0-9_]", "_", column)
        match = re.match(r"(^[0-9_]+)", no_specials)
        if match:
            number = match.groups()[0]
            rename_dict[column] = no_specials.removeprefix(number) + number
            continue
        rename_dict[column] = no_specials
    return rename_dict


#: Unused?
def check_fields_match(featurelayer, new_dataframe):
    """Make sure new data doesn't have any extra fields, warn if it doesn't contain all live fields

    Args:
        featurelayer (arcgis.features.FeatureLayer): Live data
        new_dataframe (pd.DataFrame): New data

    Raises:
        RuntimeError: If new data contains a field not present in the live data
    """

    live_fields = {field["name"] for field in featurelayer.properties.fields}
    new_fields = set(new_dataframe.columns)
    #: Remove SHAPE field from set (live "featurelayer.properties['fields']" does not expose the 'SHAPE' field)
    try:
        new_fields.remove("SHAPE")
    except KeyError:
        pass
    new_dif = new_fields - live_fields
    live_dif = live_fields - new_fields
    if new_dif:
        raise RuntimeError(
            f"New dataset contains the following fields that are not present in the live dataset: {new_dif}"
        )
    if live_dif:
        module_logger.warning(
            "New dataset does not contain the following fields that are present in the live dataset: %s", live_dif
        )


#: Unused?
def check_index_column_in_feature_layer(featurelayer, index_column):
    """Ensure index_column is present for any future operations

    Args:
        featurelayer (arcgis.features.FeatureLayer): The live feature layer
        index_column (str): The index column meant to link new and live data

    Raises:
        RuntimeError: If index_column is not in featurelayer's fields
    """

    featurelayer_fields = [field["name"] for field in featurelayer.properties.fields]
    if index_column not in featurelayer_fields:
        raise RuntimeError(f"Index column {index_column} not found in feature layer fields {featurelayer_fields}")


#: unused?
def rename_fields(dataframe, field_mapping):
    """Rename fields based on field_mapping

    Args:
        dataframe (pd.DataFrame): Dataframe with columns to be renamed
        field_mapping (dict): Mapping of existing field names to new names

    Raises:
        ValueError: If an existing name from field_mapping is not found in dataframe.columns

    Returns:
        pd.DataFrame: Dataframe with renamed fields
    """

    for original_name in field_mapping.keys():
        if original_name not in dataframe.columns:
            raise ValueError(f"Field {original_name} not found in dataframe.")

    renamed_df = dataframe.rename(columns=field_mapping)

    return renamed_df


#: This isn't used anymore... but it feels like a shame to lose it.
def build_sql_in_list(series):
    """Generate a properly formatted list to be a target for a SQL 'IN' clause

    Args:
        series (pd.Series): Series of values to be included in the 'IN' list

    Returns:
        str: Values formatted as (1, 2, 3) for numbers or ('a', 'b', 'c') for anything else
    """
    if pd.api.types.is_numeric_dtype(series):
        return f"({', '.join(series.astype(str))})"
    else:
        quoted_values = [f"'{value}'" for value in series]
        return f"({', '.join(quoted_values)})"


#: Unused in v3, but keeping for "unique constraint" info.
def check_field_set_to_unique(featurelayer, field_name):
    """Makes sure field_name has a "unique constraint" in AGOL, which allows it to be used for .append upserts

    Args:
        featurelayer (arcgis.features.FeatureLayer): The target feature layer
        field_name (str): The AGOL-valid field name to check

    Raises:
        RuntimeError: If the field is not unique (or if it's indexed but not unique)
    """

    fields = [field["fields"] for field in featurelayer.properties.indexes]
    if field_name not in fields:
        raise RuntimeError(f'{field_name} does not have a "unique constraint" set within the feature layer')
    for field in featurelayer.properties.indexes:
        if field["fields"] == field_name:
            if not field["isUnique"]:
                raise RuntimeError(f'{field_name} does not have a "unique constraint" set within the feature layer')


class Geocoding:
    """Methods for geocoding an address"""

    @staticmethod
    def geocode_addr(street, zone, api_key, rate_limits, **api_args):
        """Geocode an address through the UGRC Web API geocoder

        Invalid results are returned with an x,y of 0,0, a score of 0.0, and a match address of 'No Match'

        Args:
            street (str): The street address
            zone (str): The zip code or city
            api_key (str): API key obtained from developer.mapserv.utah.gov
            rate_limits (Tuple <float>): A lower and upper bound in seconds for pausing between API calls. Defaults to
                (0.015, 0.03)
            **api_args (dict): Keyword arguments to be passed as parameters in the API GET call. The API key will be
                added to this dict.

        Returns:
            tuple[int]: The match's x coordinate, y coordinate, score, and match address
        """

        sleep(random.uniform(rate_limits[0], rate_limits[1]))
        url = f"https://api.mapserv.utah.gov/api/v1/geocode/{street}/{zone}"
        api_args["apiKey"] = api_key

        try:
            geocode_result_dict = retry(Geocoding._geocode_api_call, url, api_args)
        except Exception as error:
            module_logger.error(error)
            return (0, 0, 0.0, "No API response")

        return (
            geocode_result_dict["location"]["x"],
            geocode_result_dict["location"]["y"],
            geocode_result_dict["score"],
            geocode_result_dict["matchAddress"],
        )

    @staticmethod
    def _geocode_api_call(url, api_args):
        """Makes a requests.get call to the geocoding API.

        Meant to be called through a retry wrapper so that the RuntimeErrors get tried again a couple times before
            finally raising the error.

        Args:
            url (str): Base url for GET request
            api_args (dict): Dictionary of URL parameters

        Raises:
            RuntimeError: If the server does not return response and request.get returns a falsy object.
            RuntimeError: If the server returns a status code other than 200 or 404

        Returns:
            dict: The 'results' dictionary of the response json (location, score, and matchAddress)
        """

        response = requests.get(url, params=api_args)

        #: The server times out and doesn't respond
        if response is None:
            module_logger.debug("GET call did not return a response")
            raise RuntimeError("No response from GET; request timeout?")

        #: The point does geocode
        if response.status_code == 200:
            return response.json()["result"]

        #: The point doesn't geocode
        if response.status_code == 404:
            return {
                "location": {"x": 0, "y": 0},
                "score": 0.0,
                "matchAddress": "No Match",
            }

        #: If we haven't returned, raise an error to trigger _retry
        raise RuntimeError(f"Did not receive a valid geocoding response; status code: {response.status_code}")

    @staticmethod
    def validate_api_key(api_key):
        """Check to see if a Web API key is valid by geocoding a single, known address point

        Args:
            api_key (str): API Key

        Raises:
            RuntimeError: If there was a network or other error attempting to geocode the known point
            ValueError: If the API responds with an invalid key message
            UserWarning: If the API responds with some other abnormal result
        """

        url = "https://api.mapserv.utah.gov/api/v1/geocode/326 east south temple street/slc"

        try:
            response = retry(requests.get, url=url, params={"apiKey": api_key})
        except Exception as error:
            raise RuntimeError(
                "Could not determine key validity; check your API key and/or network connection"
            ) from error
        response_json = response.json()
        if response_json["status"] == 200:
            return
        if response_json["status"] == 400 and "Invalid API key" in response_json["message"]:
            module_logger.error(f"API key validation failed: {response_json['message']}")
            raise ValueError(f"API key validation failed: {response_json['message']}")

        warnings.warn(f"Unhandled API key validation response {response_json['status']}: {response_json['message']}")


def calc_modulus_for_reporting_interval(n, split_value=500):
    """Calculate a number that can be used as a modulus for splitting n up into 10 or 20 intervals, depending on
    split_value.

    Args:
        n (int): The number to divide into intervals
        split_value (int, optional): The point at which it should create 20 intervals instead of 10. Defaults to 500.

    Returns:
        int: Number to be used as modulus to compare to 0 in reporting code
    """

    if n <= 10:
        return 1

    if n < split_value:
        return floor(n / 10)

    return floor(n / 20)


def authorize_pygsheets(credentials):
    """Authenticate pygsheets using either a service file or google.auth.credentials.Credentials object.

    Requires either the path to a service account .json file that has access to the files in question or  a `google.
    auth.credentials.Credentials` object. Calling `google.auth.default()` in a Google Cloud Function will give you a
    tuple of a `Credentials` object and the project id. You can use this `Credentials` object to authorize pygsheets as
    the same account the Cloud Function is running under.

    Tries first to load credentials from file; if this fails tries credentials directly as a custom_credential.

    Args:
        credentials (str or google.auth.credentials.Credentials): Path to the service file OR credentials object
            obtained from google.auth.default() within a cloud function.

    Raises:
        RuntimeError: If both authorization method attempts fail

    Returns:
        pygsheets.Client: Authorized pygsheets client
    """

    try:
        return pygsheets.authorize(service_file=credentials)
    except (FileNotFoundError, TypeError) as err:
        module_logger.debug(err)
        module_logger.debug("Credentials file not found, trying as environment variable")
    try:
        return pygsheets.authorize(custom_credentials=credentials)
    except Exception as err:
        raise RuntimeError("Could not authenticate to Google API") from err


def sedf_to_gdf(dataframe):
    """Convert an Esri Spatially Enabled DataFrame to a GeoPandas GeoDataFrame

    Args:
        dataframe (pd.DataFrame.spatial): Esri spatially enabled dataframe to convert

    Returns:
        GeoPandas.DataFrame: dataframe converted to GeoDataFrame
    """

    warnings.warn(
        "sedf_to_gdf is deprecated and will be removed in a future release. Use convert_to_gdf instead.",
        DeprecationWarning,
    )

    gdf = gpd.GeoDataFrame(dataframe, geometry=dataframe.spatial.name)
    try:
        gdf.set_crs(dataframe.spatial.sr.latestWkid, inplace=True)
    except AttributeError:
        gdf.set_crs(dataframe.spatial.sr.wkid, inplace=True)

    return gdf


def convert_to_gdf(dataframe):
    """Given a dataframe, convert it to a GeoDataFrame. Non-spatially-enabled dataframes have no geometry, allowing them to be written as gdb tables.

    Args:
        dataframe (pd.DataFrame): Input dataframe, can be a regular dataframe, gdf, or sedf.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with or without geometry.
    """

    #: Already a gdf
    if isinstance(dataframe, gpd.GeoDataFrame):
        return dataframe

    #: just a normal df, convert to gdf w/o geometry (allows us to write as table to gdb)
    try:
        dataframe.spatial.geometry_type  # raises KeyError if this is a regular dataframe
    except KeyError:
        return gpd.GeoDataFrame(dataframe, geometry=None)

    #: spatially-enabled dataframe
    gdf = gpd.GeoDataFrame(dataframe, geometry=dataframe.spatial.name)
    try:
        gdf.set_crs(dataframe.spatial.sr.latestWkid, inplace=True)
    except AttributeError:
        gdf.set_crs(dataframe.spatial.sr.wkid, inplace=True)
    return gdf


def save_to_gdb(table_or_layer, directory):
    """Save a feature_layer to a gdb for safety as backup.gdb/{layer name}_{todays date}

    Args:
        feature_layer (arcgis.features.FeatureLayer): The FeatureLayer object to save to disk.
        directory (str or Path): The directory to save the data to.

    Returns:
        Path: The full path to the output file, named with the layer name and today's date.
    """

    module_logger.debug("Downloading existing data...")
    dataframe = table_or_layer.query().sdf

    if dataframe.empty:
        return f"No data to save in feature layer {table_or_layer.properties.name}"

    if table_or_layer.properties.type == "Feature Layer":
        gdf = sedf_to_gdf(dataframe)
    else:
        gdf = gpd.GeoDataFrame(dataframe)

    out_path = Path(directory, "backup.gdb")
    out_layer = f"{table_or_layer.properties.name}_{datetime.date.today().strftime('%Y_%m_%d')}"
    module_logger.debug("Saving existing data to %s", out_path)
    try:
        gdf.to_file(out_path, layer=out_layer, engine="pyogrio", driver="OpenFileGDB")
    except pyogrio.errors.DataSourceError as error:
        raise ValueError(
            f"Error writing {out_layer} to {out_path}. Verify {Path(directory)} exists and is writable."
        ) from error

    return out_path


class FieldChecker:
    """Check the fields of a new dataframe against live data. Each method will raise errors if its checks fail.
    Provides the check_fields class method to run all the checks in one call with having to create an object.
    """

    @classmethod
    def check_fields(cls, live_data_properties, new_dataframe, fields, add_oid):
        """Run all the field checks, raising errors and warnings where they fail.

        Check individual method docstrings for details and specific errors raised.

        Args:
            live_data_properties (dict): FeatureLayer.properties of live data
            new_dataframe (pd.DataFrame): New data to be checked
            fields (List[str]): Fields to check
            add_oid (bool): Add OBJECTID to fields if its not already present (for operations that are dependent on
                OBJECTID, such as upsert)
        """

        field_checker = cls(live_data_properties, new_dataframe)
        field_checker.check_fields_present(fields, add_oid=add_oid)
        field_checker.check_live_and_new_field_types_match(fields)
        field_checker.check_for_non_null_fields(fields)
        field_checker.check_field_length(fields)
        # field_checker.check_srs_wgs84()
        field_checker.check_nullable_ints_shapely()

    def __init__(self, live_data_properties, new_dataframe):
        """
        Args:
            live_data_properties (dict): FeatureLayer.properties of live data
            new_dataframe (pd.DataFrame): New data to be checked
        """

        self.live_data_properties = live_data_properties
        self.fields_dataframe = pd.DataFrame(live_data_properties.fields)
        self.new_dataframe = new_dataframe

    def check_live_and_new_field_types_match(self, fields):
        """Raise an error if the field types of the live and new data don't match.

        Uses a dictionary mapping Esri field types to pandas dtypes. If 'SHAPE' is included in the fields, it calls
        _check_geometry_types to verify the spatial types are compatible.

        Args:
            fields (List[str]): Fields to be updated

        Raises:
            ValueError: If the field types or spatial types are incompatible, the new data has multiple geometry types,
                or the new data is not a valid spatially-enabled dataframe.
            NotImplementedError: If the live data has a field that has not yet been mapped to a pandas dtype.
        """

        #: Converting dtypes to str and comparing seems to be the only way to break out into shorts and longs, singles
        #: and doubles. Otherwise, checking subclass is probably more pythonic.
        short_ints = ["uint8", "uint16", "int8", "int16"]
        long_ints = ["int", "uint32", "uint64", "int32", "int64"]

        #: Leaving the commented types here for future implementation if necessary
        esri_to_pandas_types_mapping = {
            "esriFieldTypeInteger": ["int"] + short_ints + long_ints,
            "esriFieldTypeSmallInteger": short_ints,
            "esriFieldTypeDouble": ["float", "float32", "float64"],
            "esriFieldTypeSingle": ["float32"],
            "esriFieldTypeString": ["str", "object", "string"],
            "esriFieldTypeDate": ["datetime64[ns]"],
            "esriFieldTypeGeometry": ["geometry"],
            "esriFieldTypeOID": ["int"] + short_ints + long_ints,
            #  'esriFieldTypeBlob': [],
            "esriFieldTypeGlobalID": ["str", "object", "string"],
            #  'esriFieldTypeRaster': [],
            "esriFieldTypeGUID": ["str", "object", "string"],
            #  'esriFieldTypeXML': [],
        }

        #: geometry checking gets its own function
        if "SHAPE" in fields:
            self._check_geometry_types()
            fields.remove("SHAPE")

        fields_to_check = self.fields_dataframe[self.fields_dataframe["name"].isin(fields)].set_index("name")

        invalid_fields = []
        int_fields_as_floats = []
        datetime_fields_with_timezone = []
        for field in fields:
            #: check against the str.lower to catch normal dtypes (int64) and the new, pd.NA-aware dtypes (Int64)
            new_dtype = str(self.new_dataframe[field].dtype).lower()
            live_type = fields_to_check.loc[field, "type"]

            try:
                if new_dtype not in esri_to_pandas_types_mapping[live_type]:
                    invalid_fields.append((field, live_type, str(self.new_dataframe[field].dtype)))
                if new_dtype in ["float", "float32", "float64"] and live_type in [
                    "esriFieldTypeInteger",
                    "esriFieldTypeSmallInteger",
                ]:
                    int_fields_as_floats.append(field)
                if "datetime64" in new_dtype and new_dtype != "datetime64[ns]" and live_type == "esriFieldTypeDate":
                    datetime_fields_with_timezone.append(field)
            except KeyError:
                # pylint: disable-next=raise-missing-from
                raise NotImplementedError(f'Live field "{field}" type "{live_type}" not yet mapped to a pandas dtype')

        if invalid_fields:
            if int_fields_as_floats:
                raise IntFieldAsFloatError(
                    f"Field type incompatibilities (field, live type, new type): {invalid_fields}\n"
                    "Check the following int fields for null/np.nan values and convert to panda's nullable int "
                    f"dtype: {', '.join(int_fields_as_floats)}"
                )
            if datetime_fields_with_timezone:
                raise TimezoneAwareDatetimeError(
                    f"Field type incompatibilities (field, live type, new type): {invalid_fields}\n"
                    "Check the following datetime fields for timezone aware dtypes values and convert to "
                    "timezone-naive dtypes using pd.to_datetime(df['field']).dt.tz_localize(None): "
                    f"{', '.join(datetime_fields_with_timezone)}"
                )
            raise ValueError(f"Field type incompatibilities (field, live type, new type): {invalid_fields}")

    def _check_geometry_types(self):
        """Raise an error if the live and new data geometry types are incompatible.

        Raises:
            ValueError: If the new data is not a valid spatially-enabled dataframe, has multiple geometry types, or has
                a geometry type that doesn't match the live data.
        """

        df_to_esri_geometry_mapping = {
            "point": "esriGeometryPoint",
            "multipoint": "esriGeometryMultipoint",
            "polyline": "esriGeometryPolyline",
            "linestring": "esriGeometryPolyline",  #: gdf
            "multilinestring": "esriGeometryPolyline",  #: gdf
            "polygon": "esriGeometryPolygon",
            "multipolygon": "esriGeometryPolygon",  #: gdf
            "envelope": "esriGeometryEnvelope",
        }

        if "SHAPE" not in self.new_dataframe.columns:
            raise ValueError("New dataframe does not have a SHAPE column")

        if self.new_dataframe["SHAPE"].isna().any():
            raise ValueError(
                f"New dataframe has missing geometries at index {list(self.new_dataframe[self.new_dataframe['SHAPE'].isna()].index)}"
            )

        live_geometry_type = self.live_data_properties.geometryType

        try:
            new_geometry_types = self.new_dataframe.spatial.geometry_type
        except Exception:  #: If it's not an sedf, the call to geometry_type raises a general Exception, so try gdf
            new_geometry_types = self._condense_geopandas_multi_types(self.new_dataframe.geom_type.unique())

        if len(new_geometry_types) > 1:
            raise ValueError("New dataframe has multiple geometry types")

        if df_to_esri_geometry_mapping[new_geometry_types[0].lower()] != live_geometry_type:
            raise ValueError(
                f'New dataframe geometry type "{new_geometry_types[0]}" incompatible with live geometry type "{live_geometry_type}"'
            )

    def _condense_geopandas_multi_types(self, unique_types: np.ndarray) -> np.ndarray:
        """Given a numpy array of unique geometry types from geopandas, if both a singular type and it's corresponding multi* type are present remove the singular type; ie, ["Polygon", "MultiPolygon"] becomes ["MultiPolygon"].

        Mimics promote_to_multi arg's behavior in to_file().

        Args:
            unique_types (np.ndarray): Array of unique geometry types from a GeoDataFrame (gdf.geom_type.unique())

        Returns:
            np.ndarray: Input array with any singular geometry types replaced with their multi* partner.
        """

        if len(unique_types) == 1:
            return unique_types

        if "MultiPolygon" in unique_types and "Polygon" in unique_types:
            unique_types = unique_types[unique_types != "Polygon"]
        if "MultiLineString" in unique_types and "LineString" in unique_types:
            unique_types = unique_types[unique_types != "LineString"]

        return unique_types

    def check_for_non_null_fields(self, fields):
        """Raise an error if the new data contains nulls in a field that the live data says is not nullable.

        If this error occurs, the client should use pandas fillna() method to replace NaNs/Nones with empty strings or
        appropriate nodata values.

        Args:
            fields (List[str]): Fields to check

        Raises:
            ValueError: If the new data contains nulls in a field that the live data says is not nullable and doesn't
                have a default value.
        """

        columns_with_nulls = self.new_dataframe.columns[self.new_dataframe.isna().any()].tolist()
        # fields_dataframe = pd.DataFrame(self.live_data_properties['fields'])
        non_nullable_live_columns = self.fields_dataframe[
            ~(self.fields_dataframe["nullable"]) & ~(self.fields_dataframe["defaultValue"].astype(bool))
        ]["name"].tolist()

        columns_to_check = [column for column in columns_with_nulls if column in fields]

        #: If none of the columns have nulls, we don't need to check further
        if not columns_to_check:
            return

        problem_fields = []
        for column in columns_to_check:
            if column in non_nullable_live_columns:
                problem_fields.append(column)

        if problem_fields:
            raise ValueError(
                f"The following fields cannot have null values in the live data but one or more nulls exist in the new data: {', '.join(problem_fields)}"
            )

    def check_field_length(self, fields):
        """Raise an error if a new data string value is longer than allowed in the live data.

        Args:
            fields (List[str]): Fields to check

        Raises:
            ValueError: If the string fields in the new data contain a value longer than the corresponding field in the
                live data allows.
        """

        if "length" not in self.fields_dataframe.columns:
            module_logger.debug("No fields with length property")
            return

        length_limited_fields = self.fields_dataframe[
            (self.fields_dataframe["type"].isin(["esriFieldTypeString", "esriFieldTypeGlobalID"]))
            & (self.fields_dataframe["length"].astype(bool))
        ]

        columns_to_check = length_limited_fields[length_limited_fields["name"].isin(fields)]

        for field, live_max_length in columns_to_check[["name", "length"]].to_records(index=False):
            new_data_lengths = self.new_dataframe[field].str.len()
            new_max_length = new_data_lengths.max()
            if not pd.isna(new_max_length) and new_max_length > live_max_length:
                raise ValueError(
                    f"Row {new_data_lengths.argmax()}, column {field} in new data exceeds the live data max length of {live_max_length}"
                )

    def check_fields_present(self, fields, add_oid):
        """Raise an error if the fields to be operated on aren't present in either the live or new data.

        Args:
            fields (List[str]): The fields to be operated on.
            add_oid (bool): Add OBJECTID to fields if its not already present (for operations that are dependent on
                OBJECTID, such as upsert)

        Raises:
            RuntimeError: If any of fields are not in live or new data.
        """

        live_fields = set(self.fields_dataframe["name"])
        new_fields = set(self.new_dataframe.columns)
        working_fields = set(fields)
        working_fields.discard("SHAPE")  #: The fields from the feature layer properties don't include the SHAPE field.
        if add_oid:
            working_fields.add("OBJECTID")

        live_dif = working_fields - live_fields
        new_dif = working_fields - new_fields

        error_message = []
        if live_dif:
            error_message.append(f"Fields missing in live data: {', '.join(live_dif)}")
        if new_dif:
            error_message.append(f"Fields missing in new data: {', '.join(new_dif)}")

        if error_message:
            raise RuntimeError(". ".join(error_message))

    def check_srs_wgs84(self):
        """Raise an error if the new spatial reference system isn't WGS84 as required by geojson.

        Raises:
            ValueError: If the new SRS value can't be cast to an int (please log an issue if this occurs)
            ValueError: If the new SRS value isn't 4326.
        """

        #: If we modify a spatial data frame, sometimes the .sr.wkid property/dictionary becomes {0:number} instead
        #: of {'wkid': number}
        try:
            new_srs = self.new_dataframe.spatial.sr.wkid
        except AttributeError:
            new_srs = self.new_dataframe.spatial.sr[0]

        try:
            new_srs = int(new_srs)
        except ValueError as error:
            raise ValueError("Could not cast new SRS to int") from error
        if new_srs != 4326:
            raise ValueError(
                f"New dataframe SRS {new_srs} is not wkid 4326. Reproject with appropriate transformation."
            )

    def check_nullable_ints_shapely(self):
        """Raise a warning if null values occur within nullable integer fields of the dataframe

        Apparently due to a convention within shapely, any null values in an integer field are converted to 0.

        Raises:
            UserWarning: If we're using shapely instead of arcpy, the new dataframe uses nullable int dtypes, and there
                is one or more pd.NA values within a nullable int column.
        """

        #: Only occurs if client is using shapely instead of arcpy
        if importlib.util.find_spec("arcpy"):
            return

        nullable_ints = {"Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64"}
        nullable_int_columns = [
            column for column in self.new_dataframe.columns if str(self.new_dataframe[column].dtype) in nullable_ints
        ]
        columns_with_nulls = [column for column in nullable_int_columns if self.new_dataframe[column].isnull().any()]

        if columns_with_nulls:
            warnings.warn(
                "The following columns have null values that will be replaced by 0 due to shapely conventions: "
                f"{', '.join(columns_with_nulls)}"
            )


def get_null_geometries(feature_layer_properties):
    """Generate placeholder geometries near 0, 0 with type based on provided feature layer properties dictionary.

    Args:
        feature_layer_properties (dict): .properties from a feature layer item, contains 'geometryType' key

    Raises:
        NotImplementedError: If we get a geometryType we haven't implemented a null-geometry generator for

    Returns:
        arcgis.geometry.Geometry: A geometry object of the corresponding type centered around null island.
    """

    # esri_to_sedf_geometry_mapping = {
    #     'esriGeometryPoint': 'point',
    #     'esriGeometryMultipoint': 'multipoint',
    #     'esriGeometryPolyline': 'polyline',
    #     'esriGeometryPolygon': 'polygon',
    #     'esriGeometryEnvelope': 'envelope',
    # }

    live_geometry_type = feature_layer_properties.geometryType

    if live_geometry_type == "esriGeometryPoint":
        return arcgis.geometry.Point({"x": 0, "y": 0, "spatialReference": {"wkid": 4326}}).JSON

    if live_geometry_type == "esriGeometryPolyline":
        return arcgis.geometry.Polyline(
            {"paths": [[[0, 0], [0.1, 0.1], [0.2, 0.2]]], "spatialReference": {"wkid": 4326}}
        ).JSON

    if live_geometry_type == "esriGeometryPolygon":
        return arcgis.geometry.Polygon(
            {"rings": [[[0, 0.1], [0.1, 0.1], [0.1, 0], [0, 0]]], "spatialReference": {"wkid": 4326}}
        ).JSON

    raise NotImplementedError(f"Null value generator for live geometry type {live_geometry_type} not yet implemented")


class DeleteUtils:
    """Verify Object IDs used for delete operations"""

    @staticmethod
    def check_delete_oids_are_ints(oid_list):
        """Raise an error if a list of strings can't be parsed as ints

        Args:
            oid_list (list[int]): List of Object IDs to delete

        Raises:
            TypeError: If any of the items in oid_list can't be cast to ints

        Returns:
            list[int]: oid_list converted to ints
        """

        numeric_oids = []
        bad_oids = []
        for oid in oid_list:
            try:
                numeric_oids.append(int(oid))
            except ValueError:
                bad_oids.append(oid)
        if bad_oids:
            raise TypeError(f"Couldn't convert OBJECTID(s) `{bad_oids}` to integer")
        return numeric_oids

    @staticmethod
    def check_for_empty_oid_list(oid_list, numeric_oids):
        """Raise an error if the parsed Object ID list is empty

        Args:
            oid_list (list[int]): The original list of Object IDs to delete
            numeric_oids (list[int]): The cast-to-int Object IDs

        Raises:
            ValueError: If numeric_oids is empty
        """

        if not numeric_oids:
            raise ValueError(f"No OBJECTIDs found in {oid_list}")

    @staticmethod
    def check_delete_oids_are_in_live_data(oid_string, numeric_oids, feature_layer):
        """Warn if a delete Object ID doesn't exist in the live data, return number missing

        Args:
            oid_string (str): Comma-separated string of delete Object IDs
            numeric_oids (list[int]): The parsed and cast-to-int Object IDs
            feature_layer (arcgis.features.FeatureLayer): Live FeatureLayer item

        Raises:
            UserWarning: If any of the Object IDs in numeric_oids don't exist in the live data.

        Returns:
            int: Number of Object IDs missing from live data
        """

        query_results = feature_layer.query(object_ids=oid_string, return_ids_only=True)
        query_oids = query_results["objectIds"]
        oids_not_in_layer = set(numeric_oids) - set(query_oids)

        if oids_not_in_layer:
            warnings.warn(f"OBJECTIDs {oids_not_in_layer} were not found in the live data")

        return len(oids_not_in_layer)


class Chunking:
    """Divide a dataframe into chunks to satisfy upload size requirements for append operation."""

    @staticmethod
    def _ceildiv(num, denom):
        """Perform ceiling division: 5/4 = 2

        Args:
            num (int or float): Numerator
            denom (int or float): Denominator

        Returns:
            int: Ceiling divisor
        """

        return -(num // -denom)

    @staticmethod
    def _chunk_dataframe(dataframe, chunk_size):
        """Divide up a dataframe into a list of dataframes of chunk_size rows

        The DataFrames are returned in a list. Elements [:-1] are as large as possible for the number of chunks needed,
        while the last gets however many rows of the dataframe are left over. eg, a 10-row dataframe broken into 3
        chunks would result in dataframes with 3, 3, and 1 rows.

        Args:
            dataframe (pd.DataFrame): Input DataFrame
            chunk_size (int): The max number of rows for each sub dataframe

        Raises:
            ValueError: If the dataframe has only a single row and thus can't be chunked smaller

        Returns:
            list[pd.DataFrame]: A list of dataframes with at most chunk_size rows per dataframe
        """

        df_length = len(dataframe)

        if df_length == 1:
            raise ValueError(
                f"Dataframe chunk is only one row (index {dataframe.index[0]}), further chunking impossible"
            )

        starts = range(0, df_length, chunk_size)
        ends = [start + chunk_size if start + chunk_size < df_length else df_length for start in starts]
        list_of_dataframes = [dataframe.iloc[start:end] for start, end in zip(starts, ends)]

        return list_of_dataframes

    @staticmethod
    def build_upload_json(dataframe, feature_layer_fields, max_bytes=100_000_000):
        """Create list of geojson strings of spatially-enabled DataFrame, divided into chunks if it exceeds max_bytes

        Recursively chunks dataframe to ensure no one chunk is larger than max_bytes. Converts all empty strings in
        nullable numeric fields in feature sets created from individual chunks to None prior to converting to geojson to
        ensure the field stays numeric.

        Args:
            dataframe (pd.DataFrame.spatial): Spatially-enabled dataframe to be converted to geojson
            feature_layer_fields: All the fields from the feature layer (feature_layer.properties.fields)
            max_bytes (int, optional): Maximum size in bytes any one geojson string can be. Defaults to 100000000 (AGOL
                text uploads are limited to 100 MB?)

        Returns:
            list[str]: A list of the dataframe chunks converted to geojson
        """

        geojson_size = sys.getsizeof(dataframe.spatial.to_featureset().to_geojson.encode("utf-16"))
        module_logger.debug("Initial file size: %s", geojson_size)

        chunked_dataframes = Chunking._recursive_dataframe_chunking(dataframe, max_bytes)

        chunked_geojsons = [
            fix_numeric_empty_strings(chunk.spatial.to_featureset(), feature_layer_fields).to_geojson
            for chunk in chunked_dataframes
        ]

        return chunked_geojsons

    @staticmethod
    def _recursive_dataframe_chunking(dataframe, max_bytes):
        """Break a dataframe into chunks such that their utf-16 encoded geojson sizes don't exceed max_bytes

        Divides the dataframe into chunks based on the geojson representation's utf-16-encoded size by calculating the
        number of chunks of size > max_bytes needed for the entire file size. It uses this number of chunks to chunk the
        dataframe based on rows. Because there can be variability in geojson file size due to attribute lengths
        (especially line and polygon geometry sizes), it uses recursion to again chunk each smaller dataframe if needed.

        The chunks should (but not definitely proven to) maintain the sequential order of the features of the original
        dataframe. Suppose an initial 10 rows gives us chunks for rows [1, 2, 3], [4, 5, 6], [7, 8, 9], [10]. However,
        the second chunk [4, 5, 6] turns out to be too large, so it gets divided into [4, 5] and [6]. The resulting
        list of chunks should be [1, 2, 3], [4, 5], [6], [7, 8, 9], [10].

        The chunking process will raise an error if it tries to chunk a dataframe with only one row, which means a
        single row is larger than max_bytes (usually caused by a large and complex geometry).

        Args:
            dataframe (pd.DataFrame.spatial): A spatially-enabled dataframe to divide
            max_bytes (int): The max utf-16 encoded geojson size for any one chunk
        """

        #: Calculate number of chunks needed and the guesstimate max number of rows to achieve that size
        geojson_size = sys.getsizeof(dataframe.spatial.to_featureset().to_geojson.encode("utf-16"))
        chunks_needed = Chunking._ceildiv(geojson_size, max_bytes)
        max_rows = Chunking._ceildiv(len(dataframe), chunks_needed)

        #: Chunk the dataframe and then check if the resulting chunks are now within the proper size, calling again on
        #: the offending chunks if not
        list_of_dataframes = Chunking._chunk_dataframe(dataframe, max_rows)
        return_dataframes = []  #: Holds result of valid and recursive chunks
        for chunk_dataframe in list_of_dataframes:
            chunk_geojson_size = sys.getsizeof(chunk_dataframe.spatial.to_featureset().to_geojson.encode("utf-16"))
            if chunk_geojson_size > max_bytes:
                return_dataframes.extend(Chunking._recursive_dataframe_chunking(chunk_dataframe, max_bytes))
            else:
                return_dataframes.append(chunk_dataframe)

        return return_dataframes


def fix_numeric_empty_strings(feature_set, feature_layer_fields):
    """Replace empty strings with None for numeric fields that allow nulls
    Args:
        feature_set (arcgis.features.FeatureSet): Feature set to clean
        fields (Dict): fields from feature layer
    """

    fields_to_fix = {
        field["name"]
        for field in feature_layer_fields
        if field["type"] in ["esriFieldTypeDouble", "esriFieldTypeInteger", "esriFieldTypeDate"] and field["nullable"]
    }
    fields_to_fix -= {"Shape__Length", "Shape__Area"}

    for feature in feature_set.features:
        for field_name in fields_to_fix:
            if feature.attributes[field_name] == "":
                feature.attributes[field_name] = None

    return feature_set


def chunker(sequence, chunk_size):
    """Break sequence into chunk_size chunks

    Args:
        sequence (iterable): Any iterable sequence
        chunk_size (int): Desired number of elements in each chunk

    Returns:
        generator: Generator of original sequence broken into chunk_size lists
    """

    return (sequence[position : position + chunk_size] for position in range(0, len(sequence), chunk_size))


def is_running_in_gcp() -> bool:
    """Check if the code is running in a GCP environment

    Returns:
        bool: True if running in a GCP environment, False otherwise
    """

    #: check if the metadata server is available
    try:
        requests.get("http://metadata.google.internal", headers={"Metadata-Flavor": "Google"}, timeout=3)
    except requests.exceptions.ConnectionError:
        return False

    return True
