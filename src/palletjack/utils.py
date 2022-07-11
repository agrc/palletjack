import logging
import re
from time import sleep

import pandas as pd
import requests

module_logger = logging.getLogger(__name__)


def retry(worker_method, *args, **kwargs):
    """Allows you to retry a function/method three times to overcome network jitters

    Retries worker_method three times (for a total of four tries, including the initial attempt), pausing 2^trycount seconds between each retry. Any arguments for worker_method can be passed in as additional parameters to _retry() following worker_method: _retry(foo_method, arg1, arg2, keyword_arg=3)

    Args:
        worker_method (callable): The name of the method to be retried (minus the calling parens)

    Raises:
        error: The final error the causes worker_method to fail after 3 retries

    Returns:
        various: The value(s) returned by worked_method
    """
    tries = 1
    max_tries = 3
    delay = 2  #: in seconds

    #: this inner function (closure? almost-closure?) allows us to keep track of tries without passing it as an arg
    def _inner_retry(worker_method, *args, **kwargs):
        nonlocal tries

        try:
            return worker_method(*args, **kwargs)

        #: ArcGIS API for Python loves throwing bog-standard Exceptions, so we can't narrow this down further
        except Exception as error:
            if tries <= max_tries:  #pylint: disable=no-else-return
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
        rename_dict[column] = re.sub(r'[^a-zA-Z0-9_]', '_', column)
    return rename_dict


def replace_nan_series_with_empty_strings(dataframe):
    """Fill all completely empty series with empty strings ('')

    As of arcgis 2.0.1. to_featureset() doesn't handle completely empty series properly (it relies on str;
    https://github.com/Esri/arcgis-python-api/issues/1281), so we convert to empty strings for the time being.

    Args:
        dataframe (pd.DataFrame): Data to clean/fix

    Returns:
        pd.DataFrame: The cleaned data
    """

    for column in dataframe:
        if dataframe[column].isnull().all():
            module_logger.debug('Column %s is empty; replacing np.nans with empty strings', column)
            dataframe[column].fillna(value='', inplace=True)
    return dataframe


def check_fields_match(featurelayer, new_dataframe):
    """Make sure new data doesn't have any extra fields, warn if it doesn't contain all live fields

    Args:
        featurelayer (arcgis.features.FeatureLayer): Live data
        new_dataframe (pd.DataFrame): New data

    Raises:
        RuntimeError: If new data contains a field not present in the live data
    """

    live_fields = {field['name'] for field in featurelayer.properties['fields']}
    new_fields = set(new_dataframe.columns)
    #: Remove SHAPE field from set (live "featurelayer.properties['fields']" does not expose the 'SHAPE' field)
    try:
        new_fields.remove('SHAPE')
    except KeyError:
        pass
    new_dif = new_fields - live_fields
    live_dif = live_fields - new_fields
    if new_dif:
        raise RuntimeError(
            f'New dataset contains the following fields that are not present in the live dataset: {new_dif}'
        )
    if live_dif:
        module_logger.warning(
            'New dataset does not contain the following fields that are present in the live dataset: %s', live_dif
        )


def check_index_column_in_feature_layer(featurelayer, index_column):
    """Ensure index_column is present for any future operations

    Args:
        featurelayer (arcgis.features.FeatureLayer): The live feature layer
        index_column (str): The index column meant to link new and live data

    Raises:
        RuntimeError: If index_column is not in featurelayer's fields
    """

    featurelayer_fields = [field['name'] for field in featurelayer.properties['fields']]
    if index_column not in featurelayer_fields:
        raise RuntimeError(f'Index column {index_column} not found in feature layer fields {featurelayer_fields}')


#: This isn't used anymore... but it feels like a shame to lose it.
def build_sql_in_list(series):
    """Generate a properly formatted list to be a target for a SQL 'IN' clause

    Args:
        series (pd.Series): Series of values to be included in the 'IN' list

    Returns:
        str: Values formatted as (1, 2, 3) for numbers or ('a', 'b', 'c') for anything else
    """
    if pd.api.types.is_numeric_dtype(series):
        return f'({", ".join(series.astype(str))})'
    else:
        quoted_values = [f"'{value}'" for value in series]
        return f'({", ".join(quoted_values)})'


def check_field_set_to_unique(featurelayer, field_name):
    """Makes sure field_name has a "unique constraint" in AGOL, which allows it to be used for .append upserts

    Args:
        featurelayer (arcgis.features.FeatureLayer): The target feature layer
        field_name (str): The AGOL-valid field name to check

    Raises:
        RuntimeError: If the field is not unique (or if it's indexed but not unique)
    """

    fields = [field['fields'] for field in featurelayer.properties['indexes']]
    if field_name not in fields:
        raise RuntimeError(f'{field_name} does not have a "unique constraint" set within the feature layer')
    for field in featurelayer.properties['indexes']:
        if field['fields'] == field_name:
            if not field['isUnique']:
                raise RuntimeError(f'{field_name} does not have a "unique constraint" set within the feature layer')


def geocode_addr(row, street_col, zone_col, api_key, **api_args):
    """Geocode an address through the UGRC Web API geocoder

    Invalid results are returned as 0,0

    Args:
        row (pd.Series or dict): The row of a dataframe (or a dictionary) containing the address
        street_col (str): The column/key containing the street address
        zone_col (str): The column/key containing the zip code or city
        api_key (str): API key obtained from developer.mapserv.utah.gov
        **api_args (dict): Keyword arguments to be passed as parameters in the API GET call. The API key will be added to this dict.

    Returns:
        List<int>: The address' x and y coordinates
    """

    url = f'https://api.mapserv.utah.gov/api/v1/geocode/{row[street_col]}/{row[zone_col]}'
    api_args['apiKey'] = api_key
    r = requests.get(url, params=api_args)

    #: If we don't get a good geocode, return Null Island points so that the conversion to spatial dataframe works
    if r.status_code != 200:
        return [0, 0]
    response = r.json()
    if response['status'] != 200:
        return [0, 0]

    return list(response['result']['location'].values())
