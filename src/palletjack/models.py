"""Objects for bringing data from SFTP-hosted CSVs into AGOL Feature Services
"""

import json
import logging
from pathlib import Path

import arcgis
import numpy as np
import pandas as pd
import pysftp
from arcgis.features import GeoAccessor, GeoSeriesAccessor

logger = logging.getLogger(__name__)


class SFTPLoader:
    """Loads data from an SFTP share into a pandas DataFrame
    """

    def __init__(self, host, username, password, knownhosts_file, download_dir):
        self.host = host
        self.username = username
        self.password = password
        self.knownhosts_file = knownhosts_file
        self.download_dir = download_dir
        self._class_logger = logging.getLogger(__name__).getChild(self.__class__.__name__)

    def download_sftp_folder_contents(self, sftp_folder='upload'):
        """Download all files in sftp_folder to the SFTPLoader's download_dir

        Args:
            sftp_folder (str, optional): Path of remote folder, relative to sftp home directory. Defaults to 'upload'.
        """

        self._class_logger.info('Downloading files from `%s:%s` to `%s`', self.host, sftp_folder, self.download_dir)
        starting_file_count = len(list(self.download_dir.iterdir()))
        self._class_logger.debug('SFTP Username: %s', self.username)
        connection_opts = pysftp.CnOpts(knownhosts=self.knownhosts_file)
        with pysftp.Connection(
            self.host, username=self.username, password=self.password, cnopts=connection_opts
        ) as sftp:
            try:
                sftp.get_d(sftp_folder, self.download_dir, preserve_mtime=True)
            except FileNotFoundError as error:
                raise FileNotFoundError(f'Folder `{sftp_folder}` not found on SFTP server') from error
        downloaded_file_count = len(list(self.download_dir.iterdir())) - starting_file_count
        if not downloaded_file_count:
            raise ValueError('No files downloaded')
        return downloaded_file_count

    def download_sftp_single_file(self, filename, sftp_folder='upload'):
        """Download filename into SFTPLoader's download_dir

        Args:
            filename (str): Filename to download; used as output filename as well.
            sftp_folder (str, optional): Path of remote folder, relative to sftp home directory. Defaults to 'upload'.

        Raises:
            FileNotFoundError: Will warn if pysftp can't find the file or folder on the sftp server

        Returns:
            Path: Downloaded file's path
        """

        outfile = Path(self.download_dir, filename)

        self._class_logger.info('Downloading %s from `%s:%s` to `%s`', filename, self.host, sftp_folder, outfile)
        self._class_logger.debug('SFTP Username: %s', self.username)
        connection_opts = pysftp.CnOpts(knownhosts=self.knownhosts_file)
        try:
            with pysftp.Connection(
                self.host,
                username=self.username,
                password=self.password,
                cnopts=connection_opts,
                default_path=sftp_folder,
            ) as sftp:
                sftp.get(filename, localpath=outfile, preserve_mtime=True)
        except FileNotFoundError as error:
            raise FileNotFoundError(f'File `{filename}` or folder `{sftp_folder}`` not found on SFTP server') from error

        return outfile

    def read_csv_into_dataframe(self, filename, column_types=None):
        """Read filename into a dataframe with optional column names and types

        Args:
            filename (str): Name of file in the SFTPLoader's download_dir
            column_types (dict, optional): Column names and their dtypes(np.float64, str, etc). Defaults to None.

        Returns:
            pd.DataFrame: CSV as a pandas dataframe
        """

        self._class_logger.info('Reading `%s` into dataframe', filename)
        filepath = Path(self.download_dir, filename)
        column_names = None
        if column_types:
            column_names = list(column_types.keys())
        dataframe = pd.read_csv(filepath, names=column_names, dtype=column_types)
        self._class_logger.debug('Dataframe shape: %s', dataframe.shape)
        if len(dataframe.index) == 0:
            self._class_logger.warning('Dataframe contains no rows. Shape: %s', dataframe.shape)
        return dataframe


class FeatureServiceInlineUpdater:
    """Updates an AGOL Feature Service with data from a pandas DataFrame
    """

    def __init__(self, gis, dataframe, index_columnn):
        self.gis = gis
        self.new_dataframe = dataframe
        self.index_column = index_columnn
        self.new_data_as_dict = self.new_dataframe.set_index(index_columnn).to_dict('index')
        self._class_logger = logging.getLogger(__name__).getChild(self.__class__.__name__)

    def update_existing_features_in_feature_service_with_arcpy(self, feature_service_url, fields):
        """Update a feature service in-place with data from pandas data frame using arcpy's UpdateCursor

        Args:
            feature_service_url (str): URL to feature service
            fields (list): Names of fields to update
        """

        #: Put this here to enable the package to install/run without arcpy installed if desired
        try:
            import arcpy  # pylint: disable=import-outside-toplevel
        except ImportError as error:
            raise ImportError('Failure importing arcpy. ArcGIS Pro must be installed.') from error

        self._class_logger.info('Updating `%s` in-place', feature_service_url)
        self._class_logger.debug('Updating fields %s', fields)
        rows_updated = 0
        with arcpy.da.UpdateCursor(feature_service_url, fields) as update_cursor:
            for row in update_cursor:
                self._class_logger.debug('Evaluating row: %s', row)
                key = row[0]
                if key in self.new_data_as_dict:
                    row[1:] = list(self.new_data_as_dict[key].values())
                    self._class_logger.debug('Updating row: %s', row)
                    try:
                        update_cursor.updateRow(row)
                        rows_updated += 1
                    except RuntimeError as error:
                        if 'The value type is incompatible with the field type' in str(error):
                            raise ValueError('Field type mistmatch between dataframe and feature service') from error
        self._class_logger.info('%s rows updated', rows_updated)

        return rows_updated

    def _get_common_rows(self, live_dataframe) -> pd.DataFrame:
        """Create a dataframe containing only the rows common to both the existing live dataset and the new updates

        Args:
            live_dataframe (pd.Dataframe): The existing, live dataframe created from an AGOL hosted feature layer.

        Returns:
            pd.DataFrame: A new dataframe containing the common rows created by .merge()ing.
        """

        joined_dataframe = live_dataframe.merge(self.new_dataframe, on=[self.index_column], how='outer', indicator=True)
        subset_dataframe = joined_dataframe[joined_dataframe['_merge'] == 'both'].copy()

        not_found_dataframe = joined_dataframe[joined_dataframe['_merge'] == 'right_only'].copy()
        if not not_found_dataframe.empty:
            keys_not_found = list(not_found_dataframe[self.index_column])
            self._class_logger.warning(
                'The following keys from the new data were not found in the existing dataset: %s', keys_not_found
            )

        return subset_dataframe

    def _clean_dataframe_columns(self, dataframe, fields) -> pd.DataFrame:
        """Delete superfluous fields from dataframe that will be used for the .edit() operation

        Removes or renames the '_x', '_y' fields that are created by dataframe.merge() as appropriate based on the
        fields provided.

        Args:
            dataframe (pd.Dataframe): Dataframe to be cleaned
            fields (list[str]): The fields to keep; all others will be deleted

        Returns:
            pd.DataFrame: A dataframe with only our desired columns.
        """

        rename_dict = {f'{f}_y': f for f in fields}
        renamed_dataframe = dataframe.rename(columns=rename_dict).copy()
        fields_to_delete = [f for f in renamed_dataframe.columns if f.endswith(('_x', '_y'))]
        fields_to_delete.append('_merge')
        self._class_logger.debug('Deleting joined dataframe fields: %s', fields_to_delete)
        return renamed_dataframe.drop(labels=fields_to_delete, axis='columns', errors='ignore').copy()

    def _get_old_and_new_values(self, live_dict, object_ids):
        """Create a dictionary of the old (existing, live) and new values based on a list of object ids

        Args:
            live_dict (dict): The old (existing, live) data, key is an arbitrary index, content is another dict keyed
            by the field names.
            object_ids (list[int]): The object ids to include in the new dictionary

        Returns:
            dict: {object_id:
                    {'old_values': {data as dict from live_dict},
                     'new_values': {data as dict from self.new_dataframe}
                    },
                  ...,
                  }
        """
        oid_and_key_lookup = {
            row['OBJECTID']: row[self.index_column] for _, row in live_dict.items() if row['OBJECTID'] in object_ids
        }
        old_values_by_oid = {row['OBJECTID']: row for _, row in live_dict.items() if row['OBJECTID'] in object_ids}
        new_data_as_dict_preserve_key = self.new_dataframe.set_index(self.index_column, drop=False).to_dict('index')
        new_values_by_oid = {
            object_id: new_data_as_dict_preserve_key[key] for object_id, key in oid_and_key_lookup.items()
        }

        combined_values_by_oid = {
            object_id: {
                'old_values': old_values_by_oid[object_id],
                'new_values': new_values_by_oid[object_id]
            } for object_id in oid_and_key_lookup
        }

        return combined_values_by_oid

    def _parse_results(self, results_dict, live_dataframe) -> int:
        """Integrate .edit() results with data for reporting purposes. Relies on _get_old_and_new_values().

        Args:
            results_dict (dict): AGOL response as a python dict (raw output from .edit_features(). Defined in
            https://developers.arcgis.com/rest/services-reference/enterprise/apply-edits-feature-service-layer-.htm,
            where `true`/`false` are python True/False.
            live_dataframe (pd.DataFrame): Existing/live dataframe created from hosted feature layer.

        Returns:
            int: Number of edits successfully applied. If any failed, rollback will cause this to be set to 0.
        """

        live_dataframe_as_dict = live_dataframe.to_dict('index')
        update_results = results_dict['updateResults']
        if not update_results:
            self._class_logger.info('No update results returned; no updates attempted')
            return 0
        update_successes = [result['objectId'] for result in update_results if result['success']]
        update_failures = [result['objectId'] for result in update_results if not result['success']]
        rows_updated = len(update_successes)
        if update_successes:
            self._class_logger.info('%s rows successfully updated:', len(update_successes))
            for _, data in self._get_old_and_new_values(live_dataframe_as_dict, update_successes).items():
                self._class_logger.debug('Existing data: %s', data['old_values'])
                self._class_logger.debug('New data: %s', data['new_values'])
        if update_failures:
            self._class_logger.warning(
                'The following %s updates failed. As a result, all successfull updates should have been rolled back.',
                len(update_failures)
            )
            for _, data in self._get_old_and_new_values(live_dataframe_as_dict, update_failures).items():
                self._class_logger.warning('Existing data: %s', data['old_values'])
                self._class_logger.warning('New data: %s', data['new_values'])
            rows_updated = 0

        return rows_updated

    def update_existing_features_in_hosted_feature_layer(self, feature_layer_itemid, fields) -> int:
        """Update existing features with new attribute data in the defined fields using arcgis instead of arcpy.

        Relies on new data from self.new_dataframe. Uses the ArcGIS API for Python's .edit_features() method on a
        hosted feature layer item. May be fragile for large datasets, per .edit_features() documentation. Can't get
        .append() working properly.

        Args:
            feature_layer_itemid (str): The AGOL item id of the hosted feature layer to update.
            fields (list[str]): The field names in the feature layer to update.

        Returns:
            int: Number of features successfully updated (any failures will cause rollback and should return 0)
        """

        self._class_logger.info('Updating itemid `%s` in-place', feature_layer_itemid)
        self._class_logger.debug('Updating fields %s', fields)
        feature_layer_item = self.gis.content.get(feature_layer_itemid)
        feature_layer = arcgis.features.FeatureLayer.fromitem(feature_layer_item)
        live_dataframe = pd.DataFrame.spatial.from_layer(feature_layer)
        subset_dataframe = self._get_common_rows(live_dataframe)
        if subset_dataframe.empty:
            self._class_logger.warning(
                'No matching rows between live dataset and new dataset based on field `%s`', self.index_column
            )
            return 0
        cleaned_dataframe = self._clean_dataframe_columns(subset_dataframe, fields)
        results = feature_layer.edit_features(
            updates=cleaned_dataframe.spatial.to_featureset(), rollback_on_failure=True
        )
        log_fields = ['OBJECTID']
        log_fields.extend(fields)
        number_of_rows_updated = self._parse_results(results, live_dataframe[log_fields])
        return number_of_rows_updated


class FeatureServiceOverwriter:
    """Overwrites an AGOL Feature Service with data from a pandas DataFrame and a geometry source (Spatially-enabled
    Data Frame, feature class, etc)

    To be implemented as needed.
    """


class ColorRampReclassifier:
    """Updates the interval ranges on a layer's classification renderer based on the layer's current data.
    """

    def __init__(self, webmap_item, gis):
        self.webmap_item = webmap_item
        self.gis = gis
        self._class_logger = logging.getLogger(__name__).getChild(self.__class__.__name__)

    def _get_layer_dataframe(self, layer_name, feature_layer_number=0):
        """Create a dataframe from layer_name in self.webmap_item

        Args:
            layer_name (str): The exact name of the layer
            feature_layer_number (int): The number of the layer with the feature service to update. Defaults to 0.

        Returns:
            spatially-enabled data frame: The layer's data, including geometries.
        """

        self._class_logger.info('Getting dataframe from `%s` on `%s`', layer_name, self.webmap_item.title)
        webmap_object = arcgis.mapping.WebMap(self.webmap_item)
        layer = webmap_object.get_layer(title=layer_name)
        feature_layer = self.gis.content.get(layer['itemId'])
        layer_dataframe = pd.DataFrame.spatial.from_layer(feature_layer.layers[feature_layer_number])

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
                self._class_logger.debug('Layer `%s` has id `%s`', layer_name, layer_id)
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

        if column not in dataframe.columns:
            raise ValueError(f'Column `{column}` not in dataframe')
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
        self._class_logger.info(
            'Updating stop values on layer number `%s` in `%s`', layer_number, self.webmap_item.title
        )
        result = self.webmap_item.update(item_properties={'text': json.dumps(data)})
        self._class_logger.debug('Update result: %s', result)

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
