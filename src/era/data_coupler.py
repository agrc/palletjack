"""Objects for bringing data from SFTP-hosted CSVs into AGOL Feature Services
"""

import json
from pathlib import Path

import arcgis
import arcpy
import numpy as np
import pandas as pd
import pysftp
from arcgis import GeoAccessor, GeoSeriesAccessor


class SFTPLoader:
    """Loads data from an SFTP share into a pandas DataFrame
    """

    def __init__(self, secrets, download_dir):
        self.secrets = secrets
        self.download_dir = download_dir

    def download_sftp_files(self, sftp_folder='upload'):
        """Download all files in sftp_folder to the SFTPLoader's download_dir

        Args:
            sftp_folder (str, optional): Path of remote folder, relative to sftp home directory. Defaults to 'upload'.
        """

        connection_opts = pysftp.CnOpts(knownhosts=self.secrets.KNOWNHOSTS)
        with pysftp.Connection(
            self.secrets.SFTP_HOST,
            username=self.secrets.SFTP_USERNAME,
            password=self.secrets.SFTP_PASSWORD,
            cnopts=connection_opts
        ) as sftp:
            sftp.get_d(sftp_folder, self.download_dir, preserve_mtime=True)

    def read_csv_into_dataframe(self, filename, column_types=None):
        """Read filename into a dataframe with optional column names and types

        Args:
            filename (str): Name of file in the SFTPLoader's download_dir
            column_types (dict, optional): Column names and their dtypes(np.float64, str, etc). Defaults to None.

        Returns:
            pd.DataFrame: CSV as a pandas dataframe
        """

        filepath = Path(self.download_dir, filename)
        column_names = None
        if column_types:
            column_names = list(column_types.keys())
        dataframe = pd.read_csv(filepath, names=column_names, dtype=column_types)
        return dataframe


class FeatureServiceInLineUpdater:
    """Updates an AGOL Feature Service with data from a pandas DataFrame
    """

    def __init__(self, dataframe, index_column):
        self.dataframe = dataframe
        self.data_as_dict = self.dataframe.set_index(index_column).to_dict('index')

    def update_feature_service(self, feature_service_url, fields):
        """Update a feature service in-place with data from pandas data frame using UpdateCursor

        Args:
            feature_service_url (str): URL to feature service
            fields (list): Names of fields to update
        """

        with arcpy.da.UpdateCursor(feature_service_url, fields) as update_cursor:
            for row in update_cursor:
                key = row[0]
                if key in self.data_as_dict:
                    row[1:] = list(self.data_as_dict[key].values())
                    update_cursor.updateRow(row)


class FeatureServiceOverwriter:
    """Overwrites an AGOL Feature Service with data from a pandas DataFrame and a geometry source (Spatially-enabled
    Data Frame, feature class, etc)
    """


class ColorRampReclassifier:
    """Updates the interval ranges on a layer's classification renderer based on the layer's current data.
    """

    def __init__(self, webmap_item, gis):
        self.webmap_item = webmap_item
        self.gis = gis

    def _get_layer_dataframe(self, layer_name):
        """Create a dataframe from layer_name in self.webmap_item

        Args:
            layer_name (str): The exact name of the layer

        Returns:
            spatially-enabled data frame: The layer's data, including geometries.
        """

        webmap_object = arcgis.mapping.WebMap(self.webmap_item)
        layer = webmap_object.get_layer(title=layer_name)
        feature_layer = self.gis.content.get(layer['itemId'])
        layer_dataframe = pd.DataFrame.spatial.from_layer(feature_layer)

        return layer_dataframe

    def _get_layer_id(self, layer_name):
        """Get the ID number of layer_name in self.webmap_item

        Args:
            layer_name (str): The exact name of the layer

        Raises:
            ValueError: If the layer is not found in the webmap

        Returns:
            int: The index (0-based) of the the layer in the web map
        """

        data = self.webmap_item.get_data()
        for layer_id, layer in enumerate(data['operationalLayers']):
            if layer['title'] == layer_name:
                return layer_id

        #: If we haven't matched the title and returned a valid id, raise an error.
        raise ValueError(f'Could not find "{layer_name}" in {self.webmap_item.title}')

    @staticmethod
    def _calculate_new_stops(dataframe, column, stops):
        """Calculate new stop values for an AGOL color ramp using what appears to be AGOL's method for unclassed ramps.

        Args:
            dataframe (pd.DataFrame): Data being classified
            column (str): Column to classify
            stops (int, optional): Number of stops to create.

        Returns:
            List: New stops cast as ints
        """

        minval = dataframe[column].min()
        mean = dataframe[column].mean()
        std_dev = dataframe[column].std()
        upper = mean + std_dev  #: AGOL's default upper value for unclassed ramps seems to be mean + 1 std dev

        new_stops = np.linspace(minval, upper, stops)
        new_stops_ints = [int(stop) for stop in new_stops]

        return new_stops_ints

    def _update_stop_values(self, layer_number, new_stops):
        """Update the stop values of an (un)classified polygon renderer in an AGOL Webmap

        Args:
            layer_number (int): The index for the layer to be updated
            new_stops (List): New values for the existing stops

        Returns:
            Bool: Success or failure of update operation
        """

        #: Get short reference to the stops dictionary from the webmap's data json
        data = self.webmap_item.get_data()
        renderer = data['operationalLayers'][layer_number]['layerDefinition']['drawingInfo']['renderer']
        stops = renderer['visualVariables'][0]['stops']

        #: Overwrite the value, update the webmap item
        for stop, new_value in zip(stops, new_stops):
            stop['value'] = new_value
        result = self.webmap_item.update(item_properties={'text': json.dumps(data)})

        return result

    def update_color_ramp_values(self, layer_name, column_name, stops=5):
        """Update the color ramp ranges for layer_name in self.webmap_item.

        Does not alter colors or introduce additional stops; only overwrites the values for existing breaks.

        Args:
            layer_name (str): The exact name of the layer to be updated
            column_name (str): The name of the attribute being displayed as an (un)classified range
            stops (int, optional): The number of stops to calculate. Must match existing stops. Defaults to 5.

        Returns:
            Bool: Success or failure of update operation
        """
        layer_id = self._get_layer_id(layer_name)
        dataframe = self._get_layer_dataframe(layer_name)
        new_stops = self._calculate_new_stops(dataframe, column_name, stops)
        result = self._update_stop_values(layer_id, new_stops)

        return result
