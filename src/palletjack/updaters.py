"""Objects for bringing data from SFTP-hosted CSVs into AGOL Feature Services
"""

import json
import logging
import warnings
from pathlib import Path

import arcgis
import numpy as np
import pandas as pd
from arcgis.features import GeoAccessor, GeoSeriesAccessor

logger = logging.getLogger(__name__)


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

    def _validate_working_fields_in_live_and_new_dataframes(self, live_dataframe, fields):
        live_fields = set(live_dataframe.columns)
        new_fields = set(self.new_dataframe.columns)
        working_fields = set(fields)

        live_dif = working_fields - live_fields
        new_dif = working_fields - new_fields

        if live_dif or new_dif:
            raise RuntimeError(
                f'Field mismatch between defined fields and either new or live data.\nFields not in live data: {live_dif}\nFields not in new data: {new_dif}'
            )

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

        #: rename specified fields, delete any other _x/_y fields (in effect, just keep the specified fields...)
        #: TODO: There may be a simpler way to do this- maybe .reindex(fields)?
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
        self._validate_working_fields_in_live_and_new_dataframes(live_dataframe, fields)
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


class FeatureServiceAttachmentsUpdater:
    """Add or overwrite attachments in a feature service using a dataframe of the desired "new" attachments. Uses a
    join field present in both live data and the attachments dataframe to match attachments with live data.
    """

    def __init__(self, gis):
        self.gis = gis
        self._class_logger = logging.getLogger(__name__).getChild(self.__class__.__name__)
        self.failed_dict = {}

    #: This isn't used anymore... but it feels like a shame to lose it.
    @staticmethod
    def _build_sql_in_list(series):
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

    def _get_live_oid_and_guid_from_join_field_values(self, live_features_as_df, attachment_join_field, attachments_df):
        """Get the live Object ID and guid from the live features for only the features in attachments_df

        Args:
            live_features_as_df (pd.DataFrame): Spatial dataframe of all the feature layer's live data from AGOL
            attachment_join_field (str): Column in attachments_df to use as a join key with live data
            attachments_df (pd.DataFrame): New attachment data, including the join key and a path to the "new"
                                           attachment

        Returns:
            pd.DataFrame: Attachments dataframe with corresponding live OIDs and GUIDs.
        """

        self._class_logger.debug('Using %s as the join field between live and new data', attachment_join_field)
        subset_df = live_features_as_df.reindex(columns=['OBJECTID', 'GlobalID', attachment_join_field])
        merged_df = subset_df.merge(attachments_df, on=attachment_join_field, how='inner')
        self._class_logger.debug('%s features common to both live and new data', len(merged_df.index))

        return merged_df

    def _get_current_attachment_info_by_oid(self, live_data_subset_df):
        """Merge the live attachment data using the ObjectID as the join

        Args:
            live_data_subset_df (pd.DataFrame): Live data with 'OBJECTID', 'GlobalID', and new attachment data

        Returns:
            pd.DataFrame: Live and new attachment data in one dataframe
        """

        live_attachments_df = pd.DataFrame(self.feature_layer.attachments.search())
        live_attachments_subset_df = live_attachments_df.reindex(columns=['PARENTOBJECTID', 'NAME', 'ID'])
        merged_df = live_data_subset_df.merge(
            live_attachments_subset_df, left_on='OBJECTID', right_on='PARENTOBJECTID', how='left'
        )
        #: Cast ID field to nullable int to avoid conversion to float for np.nans
        merged_df['ID'] = merged_df['ID'].astype('Int64')

        return merged_df

    def _create_attachment_action_df(self, attachment_eval_df, attachment_path_field):
        """Create a dataframe containing the action needed for each feature resulting from the attachment join.

        If the live feature doesn't have an attachment, add the attachment. If it does, compare the file names and only
        attach if they are different. Otherwise, leave null.

        Args:
            attachment_eval_df (pd.DataFrame): DataFrame of live attachment data, subsetted to features that matched
                                               the join key in the new attachments
            attachment_path_field (str): The column that holds the attachment path

        Returns:
            pd.DataFrame: attachment_eval_df with 'operation' and 'new_filename' columns added
        """

        #: Get the file name from the full path
        attachment_eval_df['new_filename'] = attachment_eval_df[attachment_path_field].apply(
            lambda path: Path(path).name
        )

        #: Overwrite if different names, add if no existing name, do nothing if names are the same
        attachment_eval_df['operation'] = np.nan
        attachment_eval_df.loc[attachment_eval_df['NAME'] != attachment_eval_df['new_filename'],
                               'operation'] = 'overwrite'
        attachment_eval_df.loc[attachment_eval_df['NAME'].isna(), 'operation'] = 'add'

        value_counts = attachment_eval_df['operation'].value_counts(dropna=False)
        for operation in ['add', 'overwrite', np.nan]:
            if operation not in value_counts:
                value_counts[operation] = 0
        self._class_logger.debug(
            'Calculated attachment operations: adds: %s, overwrites: %s, none: %s', value_counts['add'],
            value_counts['overwrite'], value_counts[np.nan]
        )

        return attachment_eval_df

    def _add_attachments_by_oid(self, attachment_action_df, attachment_path_field):
        """Add attachments using the feature's OID based on the 'operation' field of the dataframe

        Args:
            attachment_action_df (pd.DataFrame): A dataframe containing 'operation', 'OBJECTID', and
                                                 attachment_path_field columns
            attachment_path_field (str): The column that holds the attachment path

        Returns:
            int: The number of features that successfully have attachments added.
        """

        adds_dict = attachment_action_df[attachment_action_df['operation'] == 'add'].to_dict(orient='index')
        adds_count = 0

        for row in adds_dict.values():
            target_oid = row['OBJECTID']
            filepath = row[attachment_path_field]

            self._class_logger.debug('Add %s to OID %s', filepath, target_oid)
            try:
                result = self.feature_layer.attachments.add(target_oid, filepath)
            except Exception:
                self._class_logger.error('AGOL error while adding %s to OID %s', filepath, target_oid, exc_info=True)
                self.failed_dict[target_oid] = ('add', filepath)
                continue

            self._class_logger.debug('%s', result)
            if not result['addAttachmentResult']['success']:
                warnings.warn(f'Failed to attach {filepath} to OID {target_oid}')
                self.failed_dict[target_oid] = ('add', filepath)
                continue

            adds_count += 1

        return adds_count

    def _overwrite_attachments_by_oid(self, attachment_action_df, attachment_path_field):
        """Overwrite attachments using the feature's OID based on the 'operation' field of the dataframe

        Args:
            attachment_action_df (pd.DataFrame): A dataframe containing 'operation', 'OBJECTID', 'ID', 'NAME', and
                                                 attachment_path_field columns
            attachment_path_field (str): The column that holds the attachment path

        Returns:
            int: The number of features that successfully have their attachments overwritten.
        """

        overwrites_dict = attachment_action_df[attachment_action_df['operation'] == 'overwrite'].to_dict(orient='index')
        overwrites_count = 0

        for row in overwrites_dict.values():
            target_oid = row['OBJECTID']
            filepath = row[attachment_path_field]
            attachment_id = row['ID']
            old_name = row['NAME']

            self._class_logger.debug(
                'Overwriting %s (attachment ID %s) on OID %s with %s', old_name, attachment_id, target_oid, filepath
            )
            try:
                result = self.feature_layer.attachments.update(target_oid, attachment_id, filepath)
            except Exception:
                self._class_logger.error(
                    'AGOL error while overwritting %s (attachment ID %s) on OID %s with %s',
                    old_name,
                    attachment_id,
                    target_oid,
                    filepath,
                    exc_info=True
                )
                self.failed_dict[target_oid] = ('update', filepath)
                continue

            self._class_logger.debug('%s', result)
            if not result['updateAttachmentResult']['success']:
                warnings.warn(
                    f'Failed to update {old_name}, attachment ID {attachment_id}, on OID {target_oid} with {filepath}'
                )
                self.failed_dict[target_oid] = ('update', filepath)
                continue

            overwrites_count += 1

        return overwrites_count

    @staticmethod
    def _check_attachment_dataframe_for_invalid_column_names(attachment_dataframe, invalid_names):
        invalid_names_index = pd.Index(invalid_names)
        intersection = attachment_dataframe.columns.intersection(invalid_names_index)
        if not intersection.empty:
            raise RuntimeError(f'Attachment dataframe contains the following invalid names: {list(intersection)}')

    def update_attachments(
        self, feature_layer_itemid, attachment_join_field, attachment_path_field, attachments_df, layer_number=0
    ):
        """Update a feature layer's attachments based on info from a dataframe of desired attachment file names

        Depends on a dataframe populated with a join key for the live data and the downloaded or locally-available
        attachments. If the name of the "new" attachment is the same as an existing attachment for that feature, it is
        not updated. If it is different or there isn't an existing attachment, the "new" attachment is attached to that
        feature.

        Args:
            feature_layer_itemid (str): The AGOL Item ID of the feature layer to update
            attachment_join_field (str): The field containing the join key between the attachments dataframe and the
                                         live data
            attachment_path_field (str): The field containing the desired attachment file path
            attachments_df (pd.DataFrame): A dataframe of desired attachments, including a join key and the local path
                                           to the attachment
            layer_number (int, optional): The layer within the Item ID to update. Defaults to 0.

        Returns:
            (int, int): Tuple of counts of successful overwrites and adds.
        """

        self._class_logger.info('Updating attachments...')
        #: These names are present in the live attachment data downloaded from AGOL. Because we merge the dataframes
        #: later, we need to make sure they're not the same. There may be better ways of handling this that allows the
        #: client names to be preserved, but for now force them to fix this.
        self._check_attachment_dataframe_for_invalid_column_names(
            attachments_df, invalid_names=['OBJECTID', 'PARENTOBJECTID', 'NAME', 'ID']
        )
        self._class_logger.debug('Using layer %s from item ID %s', layer_number, feature_layer_itemid)
        self.feature_layer = self.gis.content.get(feature_layer_itemid).layers[layer_number]
        live_features_as_df = pd.DataFrame.spatial.from_layer(self.feature_layer)
        live_data_subset_df = self._get_live_oid_and_guid_from_join_field_values(
            live_features_as_df, attachment_join_field, attachments_df
        )
        #: TODO: Make sure layer supports attachments so we don't get an arcgis error.
        #: Check out the feature layer .properties and FeatureLayerManager.add_to_definition to check/enable?
        attachment_eval_df = self._get_current_attachment_info_by_oid(live_data_subset_df)
        attachment_action_df = self._create_attachment_action_df(attachment_eval_df, attachment_path_field)

        overwrites_count = self._overwrite_attachments_by_oid(attachment_action_df, attachment_path_field)
        adds_count = self._add_attachments_by_oid(attachment_action_df, attachment_path_field)
        self._class_logger.info('%s attachments added, %s attachments overwritten', adds_count, overwrites_count)

        return overwrites_count, adds_count

    @staticmethod
    def build_attachments_dataframe(input_dataframe, join_column, attachment_column, out_dir):
        """Create an attachments dataframe by subsetting down to just the two fields and dropping any rows
           with null/empty attachments

        Args:
            input_dataframe (pd.DataFrame): Input data containing at least the join and attachment filename columns
            join_column (str): Unique key joining attachments to live data
            attachment_column (str): Filename for each attachment
            out_dir (str or Path): Output directory, will be used to build full path to attachment

        Returns:
            pd.DataFrame: Dataframe with join key, attachment name, and full attachment paths
        """

        #: Create an attachments dataframe by subsetting down to just the two fields and dropping any rows
        #: with null/empty attachments

        input_dataframe[attachment_column].replace('', np.nan, inplace=True)  #: pandas doesn't see empty strings as NAs
        attachments_dataframe = input_dataframe[[join_column, attachment_column]] \
                                            .copy().dropna(subset=[attachment_column])
        #: Create the full path by prepending the output directory using .apply and a lambda function
        attachments_dataframe['full_file_path'] = attachments_dataframe[attachment_column] \
                                                    .apply(lambda filename: str(Path(out_dir, filename)))

        return attachments_dataframe


class FeatureServiceOverwriter:
    """Overwrites an AGOL Feature Service with data from a pandas DataFrame and a geometry source (Spatially-enabled
    Data Frame, feature class, etc)

    To be implemented as needed.
    """


#: TODO: implement for Rick's fleet stuff


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
