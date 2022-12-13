"""Objects for bringing data from SFTP-hosted CSVs into AGOL Feature Services
"""

import datetime
import json
import logging
import warnings
from pathlib import Path

import arcgis
import numpy as np
import pandas as pd
from arcgis.features import GeoAccessor, GeoSeriesAccessor

from . import transform, utils

logger = logging.getLogger(__name__)


class FeatureServiceUpdater:
    """Updates an AGOL Feature Service with data from a pandas DataFrame
    """

    @classmethod
    def add_data(cls, gis, feature_service_itemid, dataframe, layer_index=0):
        """Adds new features to existing hosted feature layer. Uses fields from new dataframe.

        Raises:
            ValueError: If the new field and existing fields don't match, the new data contains null fields,
                        the new data exceeds the existing field lengths, or a specified field is missing from either
                        new or live data.

        Returns:
            int: Number of features added
        """
        updater = cls(gis, feature_service_itemid, dataframe, fields=list(dataframe.columns), layer_index=layer_index)
        return updater._add_new_data_to_hosted_feature_layer()

    @classmethod
    def remove_data(cls, gis, feature_service_itemid, dataframe, join_column, layer_index=0):
        pass

    @classmethod
    def update_data(cls, gis, feature_service_itemid, dataframe, join_column, layer_index=0):
        """Updates existing features within a hosted feature layer using OBJECTID as the join field

        Raises:
            ValueError: If the new field and existing fields don't match, the new data contains null fields,
                        the new data exceeds the existing field lengths, or a specified field is missing from either
                        new or live data.

        Returns:
            int: Number of features updated
        """
        # updater = cls(
        #     gis, feature_service_itemid, dataframe, join_column=join_column, fields=fields, layer_index=layer_index
        # )
        updater = cls(
            gis,
            feature_service_itemid,
            dataframe,
            join_column=join_column,
            fields=list(dataframe.columns),
            layer_index=layer_index
        )
        return updater._update_hosted_feature_layer()

    @classmethod
    def overwrite_data(cls, gis, feature_service_itemid, dataframe, failsafe_dir, layer_index=0):
        pass

    def __init__(
        self, gis, feature_service_itemid, dataframe, fields=None, join_column=None, failsafe_dir=None, layer_index=0
    ):
        self._class_logger = logging.getLogger(__name__).getChild(self.__class__.__name__)
        self.feature_service_itemid = feature_service_itemid
        self.feature_layer = arcgis.features.FeatureLayer.fromitem(gis.content.get(feature_service_itemid))
        self.new_dataframe = dataframe.rename(columns=utils.rename_columns_for_agol(dataframe.columns))
        if 'SHAPE' in self.new_dataframe.columns:
            self.new_dataframe.spatial.set_geometry('SHAPE')
        if fields:
            self.fields = list(utils.rename_columns_for_agol(fields).values())
        if join_column:
            self.join_column = utils.rename_columns_for_agol([join_column])[join_column]
        self.failsafe_dir = failsafe_dir
        self.layer_index = layer_index

    # def __init__(self, gis, dataframe, index_column, field_mapping=None):
    #     self._class_logger = logging.getLogger(__name__).getChild(self.__class__.__name__)
    #     self.gis = gis
    #     self.index_column = utils.rename_columns_for_agol([index_column])[index_column]

    #     if field_mapping:
    #         dataframe = utils.rename_fields(dataframe, field_mapping)

    #     self.new_dataframe = dataframe.rename(columns=utils.rename_columns_for_agol(dataframe.columns))
    #     try:
    #         self.new_data_as_dict = self.new_dataframe.set_index(self.index_column).to_dict('index')
    #     except KeyError as error:
    #         raise KeyError(f'Index column {index_column} not found in dataframe columns') from error

    # def _get_common_rows(self, live_dataframe) -> pd.DataFrame:
    #     """Create a dataframe containing only the rows common to both the existing live dataset and the new updates

    #     Args:
    #         live_dataframe (pd.Dataframe): The existing, live dataframe created from an AGOL hosted feature layer.

    #     Returns:
    #         pd.DataFrame: A new dataframe containing the common rows created by .merge()ing.
    #     """

    #     subset_dataframe = live_dataframe[live_dataframe[self.index_column] \
    #         .isin(self.new_dataframe[self.index_column])] \
    #         .set_index(self.index_column)
    #     subset_dataframe.update(self.new_dataframe.set_index(self.index_column))

    #     not_found_dataframe = self.new_dataframe[~self.new_dataframe[self.index_column]
    #                                              .isin(live_dataframe[self.index_column])]
    #     if not not_found_dataframe.empty:
    #         keys_not_found = list(not_found_dataframe[self.index_column])
    #         self._class_logger.warning(
    #             'The following keys from the new data were not found in the existing dataset: %s', keys_not_found
    #         )

    #     return subset_dataframe.reset_index()

    # def _clean_dataframe_columns(self, dataframe, fields) -> pd.DataFrame:
    #     """Delete superfluous fields from dataframe that will be used for the .edit() operation

    #     Removes or renames the '_x', '_y' fields that are created by dataframe.merge() as appropriate based on the
    #     fields provided.

    #     Args:
    #         dataframe (pd.Dataframe): Dataframe to be cleaned
    #         fields (list[str]): The fields to keep; all others will be deleted

    #     Returns:
    #         pd.DataFrame: A dataframe with only our desired columns.
    #     """

    #     #: rename specified fields, delete any other _x/_y fields (in effect, just keep the specified fields...)
    #     #: TODO: There may be a simpler way to do this- maybe .reindex(fields)?
    #     rename_dict = {f'{f}_y': f for f in fields}
    #     renamed_dataframe = dataframe.rename(columns=rename_dict).copy()
    #     fields_to_delete = [f for f in renamed_dataframe.columns if f.endswith(('_x', '_y'))]
    #     fields_to_delete.append('_merge')
    #     self._class_logger.debug('Deleting joined dataframe fields: %s', fields_to_delete)
    #     return renamed_dataframe.drop(labels=fields_to_delete, axis='columns', errors='ignore').copy()

    # def _get_old_and_new_values(self, live_dict, object_ids):
    #     """Create a dictionary of the old (existing, live) and new values based on a list of object ids

    #     Args:
    #         live_dict (dict): The old (existing, live) data, key is an arbitrary index, content is another dict keyed
    #         by the field names.
    #         object_ids (list[int]): The object ids to include in the new dictionary

    #     Returns:
    #         dict: {object_id:
    #                 {'old_values': {data as dict from live_dict},
    #                  'new_values': {data as dict from self.new_dataframe}
    #                 },
    #               ...,
    #               }
    #     """
    #     oid_and_key_lookup = {
    #         row['OBJECTID']: row[self.index_column] for _, row in live_dict.items() if row['OBJECTID'] in object_ids
    #     }
    #     old_values_by_oid = {row['OBJECTID']: row for _, row in live_dict.items() if row['OBJECTID'] in object_ids}
    #     new_data_as_dict_preserve_key = self.new_dataframe.set_index(self.index_column, drop=False).to_dict('index')
    #     new_values_by_oid = {
    #         object_id: new_data_as_dict_preserve_key[key] for object_id, key in oid_and_key_lookup.items()
    #     }

    #     combined_values_by_oid = {
    #         object_id: {
    #             'old_values': old_values_by_oid[object_id],
    #             'new_values': new_values_by_oid[object_id]
    #         } for object_id in oid_and_key_lookup
    #     }

    #     return combined_values_by_oid

    # def _parse_results(self, results_dict, live_dataframe) -> int:
    #     """Integrate .edit() results with data for reporting purposes. Relies on _get_old_and_new_values().

    #     Args:
    #         results_dict (dict): AGOL response as a python dict (raw output from .edit_features(). Defined in
    #         https://developers.arcgis.com/rest/services-reference/enterprise/apply-edits-feature-service-layer-.htm,
    #         where `true`/`false` are python True/False.
    #         live_dataframe (pd.DataFrame): Existing/live dataframe created from hosted feature layer.

    #     Returns:
    #         int: Number of edits successfully applied. If any failed, rollback will cause this to be set to 0.
    #     """

    #     live_dataframe_as_dict = live_dataframe.to_dict('index')
    #     update_results = results_dict['updateResults']
    #     if not update_results:
    #         self._class_logger.info('No update results returned; no updates attempted')
    #         return 0
    #     update_successes = [result['objectId'] for result in update_results if result['success']]
    #     update_failures = [result['objectId'] for result in update_results if not result['success']]
    #     rows_updated = len(update_successes)
    #     if update_successes:
    #         self._class_logger.info('%s rows successfully updated:', len(update_successes))
    #         for _, data in self._get_old_and_new_values(live_dataframe_as_dict, update_successes).items():
    #             self._class_logger.debug('Existing data: %s', data['old_values'])
    #             self._class_logger.debug('New data: %s', data['new_values'])
    #     if update_failures:
    #         self._class_logger.warning(
    #             'The following %s updates failed. As a result, all successful updates should have been rolled back.',
    #             len(update_failures)
    #         )
    #         for _, data in self._get_old_and_new_values(live_dataframe_as_dict, update_failures).items():
    #             self._class_logger.warning('Existing data: %s', data['old_values'])
    #             self._class_logger.warning('New data: %s', data['new_values'])
    #         rows_updated = 0

    #     return rows_updated

    def _add_new_data_to_hosted_feature_layer(self) -> int:
        """Adds new features to existing hosted feature layer. Uses fields from new dataframe.

        Raises:
            ValueError: If the new field and existing fields don't match, the new data contains null fields,
                        the new data exceeds the existing field lengths, or a specified field is missing from either
                        new or live data.

        Returns:
            int: Number of features added
        """

        self._class_logger.info(
            'Adding items to layer `%s` in itemid `%s` in-place', self.layer_index, self.feature_service_itemid
        )
        self._class_logger.debug('Using fields %s', self.fields)

        #: Field checks to prevent Error: 400 errors from AGOL
        field_checker = utils.FieldChecker(self.feature_layer.properties, self.new_dataframe)
        field_checker.check_live_and_new_field_types_match(self.fields)
        field_checker.check_for_non_null_fields(self.fields)
        field_checker.check_field_length(self.fields)
        field_checker.check_fields_present(self.fields, add_oid=False)

        #: Upsert
        messages = self._upsert_data(
            self.feature_layer,
            self.new_dataframe,
            upsert=False,
        )
        return messages['recordCount']

    def _update_hosted_feature_layer(self) -> int:
        """Updates existing features within a hosted feature layer using OBJECTID as the join field

        Raises:
            ValueError: If the new field and existing fields don't match, the new data contains null fields,
                        the new data exceeds the existing field lengths, or a specified field is missing from either
                        new or live data.

        Returns:
            int: Number of features updated
        """

        self._class_logger.info(
            'Updating layer `%s` in itemid `%s` in-place', self.layer_index, self.feature_service_itemid
        )
        self._class_logger.debug('Updating fields %s', self.fields)

        #: Field checks to prevent Error: 400 errors from AGOL
        field_checker = utils.FieldChecker(self.feature_layer.properties, self.new_dataframe)
        field_checker.check_live_and_new_field_types_match(self.fields)
        field_checker.check_for_non_null_fields(self.fields)
        field_checker.check_field_length(self.fields)
        field_checker.check_fields_present(self.fields, add_oid=True)

        #: Upsert
        messages = self._upsert_data(
            self.feature_layer,
            self.new_dataframe,
            upsert=True,
            upsert_matching_field='OBJECTID',
            append_fields=self.fields
        )
        return messages['recordCount']

    #: TODO: Figuring out which fields to pass to upsert. Upsert doesn't need OBJECTID (will use existing if it's not included in append_fields). to_featureset() requires a SHAPE field, so we have to get shape for updating existing data (can we pass a stand-in, simple geometry and see if it's ignored if SHAPE isn't in append_fields?). Test whether upsert_new_data requires matching field. Make sure matching field are in append_fields (either explicitly or in earlier steps?).
    #: to_featureset() will set objectid if it's not provided.
    #: TODO: join live and new dataframes by self.index_column, but append using OID/GUID
    #: TODO: Need to allow table-only (no geometry) updates, but to_geojson requires geometry. auto add null geometries and verify they don't get used.
    def _upsert_data(self, target_featurelayer, dataframe, **append_kwargs):
        """UPdate and inSERT data into live dataset with featurelayer.append()

        Note: The call to to_featureset() in this method will add new OBJECTIDs to the new data if they aren't already present. If

        Args:
            target_featurelayer (arcgis.features.FeatureLayer): Live dataset
            dataframe (pd.DataFrame): Spatially-enabled dataframe containing new data. Column names must match live
            data. New data must have a valid shape column
            append_kwargs (keyword arguments): Arguments to be passed to .append()

        Raises:
            RuntimeError: If append fails; will attempt to rollback appends that worked.

        Returns:
            dict: Messages returned from append operation
        """

        try:
            if append_kwargs['upsert'] \
                and (
                        append_kwargs['upsert_matching_field'] not in append_kwargs['append_fields']
                        or
                        append_kwargs['upsert_matching_field'] not in dataframe.columns
                    ):
                raise ValueError(
                    f'Upsert matching field {append_kwargs["upsert_matching_field"]} not found in either append fields or existing fields.'
                )
        except KeyError:
            pass

        geojson = dataframe.spatial.to_featureset().to_geojson
        result, messages = utils.retry(
            target_featurelayer.append,
            upload_format='geojson',
            edits=geojson,
            return_messages=True,
            rollback=True,
            **append_kwargs
        )

        self._class_logger.debug(messages)
        if not result:
            raise RuntimeError('Failed to append data. Append operation should have been rolled back.')

        return messages

    # def append_new_data_to_hosted_feature_layer(self, feature_service_itemid, layer_index=0):
    #     """UPdate existing data and inSERT new data to a hosted feature layer from a spatially-enabled data frame

    #     Relies on the FeatureServiceInlineUpdater's self.new_dataframe for the new data and self.index_column to define
    #     the common column used to match new and existing data. The field specified by self.index_column must have a
    #     "unique constraint" in the target feature layer (ie, it must be indexed and be unique).

    #     Args:
    #         feature_service_itemid (str): The AGOL item id for the target feature layer
    #         layer_index (int, optional): The layer id within the target feature layer. Defaults to 0.

    #     Raises:
    #         RuntimeError: If append fails; will attempt to rollback appends that worked.
    #         RuntimeError: If new data contains a field not present in the live data
    #         Warning: If live data contains a field not present in the new data
    #         RuntimeError: If index_column is not in featurelayer's fields
    #         RuntimeError: If the field is not unique (or if it's indexed but not unique)

    #     Returns:
    #         int: The number of records updated
    #     """

    #     self._class_logger.info('Updating layer `%s` in itemid `%s` in-place', layer_index, feature_service_itemid)
    #     target_featurelayer = arcgis.features.FeatureLayer.fromitem(
    #         self.gis.content.get(feature_service_itemid), layer_id=layer_index
    #     )
    #     #: temp fix until Esri fixes empty series as NaN bug
    #     fixed_dataframe = utils.replace_nan_series_with_empty_strings(self.new_dataframe)
    #     utils.check_fields_match(target_featurelayer, fixed_dataframe)
    #     # utils.check_index_column_in_feature_layer(target_featurelayer, self.index_column)
    #     # utils.check_field_set_to_unique(target_featurelayer, self.index_column)
    #     messages = self._upsert_data(target_featurelayer, fixed_dataframe, upsert=False)

    #     return messages['recordCount']


class FeatureServiceAttachmentsUpdater:
    """Add or overwrite attachments in a feature service using a dataframe of the desired "new" attachments. Uses a
    join field present in both live data and the attachments dataframe to match attachments with live data.
    """

    def __init__(self, gis):
        self.gis = gis
        self._class_logger = logging.getLogger(__name__).getChild(self.__class__.__name__)
        self.failed_dict = {}

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
                    'AGOL error while overwriting %s (attachment ID %s) on OID %s with %s',
                    old_name,
                    attachment_id,
                    target_oid,
                    filepath,
                    exc_info=True
                )
                self.failed_dict[target_oid] = ('update', filepath)
                continue

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

        input_dataframe[attachment_column].replace('', np.nan, inplace=True)  #: pandas doesn't see empty strings as NAs
        attachments_dataframe = input_dataframe[[join_column, attachment_column]] \
                                            .copy().dropna(subset=[attachment_column])
        #: Create the full path by prepending the output directory using .apply and a lambda function
        attachments_dataframe['full_file_path'] = attachments_dataframe[attachment_column] \
                                                    .apply(lambda filename: str(Path(out_dir, filename)))

        return attachments_dataframe


class FeatureServiceOverwriter:
    """Overwrites an AGOL Feature Service with data from a spatially-enabled DataFrame.
    """

    def __init__(self, gis):
        self.gis = gis
        self._class_logger = logging.getLogger(__name__).getChild(self.__class__.__name__)

    def _truncate_existing_data(self, featurelayer, layer_index, itemid):
        """Remove all existing features from live dataset

        Args:
            featurelayer (arcgis.features.FeatureLayer): Live data
            layer_index (int): Layer id within the main feature service
            itemid (str): AGOL item ID for the main feature service

        Raises:
            RuntimeError: If the truncate fails

        Returns:
            pd.DataFrame: The feature layer's data prior to truncating
        """

        old_data = featurelayer.query(as_df=True)
        truncate_result = utils.retry(featurelayer.manager.truncate, asynchronous=True, wait=True)
        self._class_logger.debug(truncate_result)
        if truncate_result['status'] != 'Completed':
            raise RuntimeError(f'Failed to truncate existing data from layer id {layer_index} in itemid {itemid}')
        return old_data

    def _append_new_data(self, target_featurelayer, dataframe, feature_service_item_id, layer_index):
        """Add new data to live dataset

        Args:
            target_featurelayer (arcgis.features.FeatureLayer): Live dataset; should be empty
            dataframe (pd.DataFrame): Spatially-enabled dataframe containing new data. Column names should match live
            data
            feature_service_item_id (int): Layer ID within the main feature service
            layer_index (str): AGOL item ID for the main feature service

        Raises:
            RuntimeError: If append fails; will attempt to rollback appends that worked.

        Returns:
            dict: Messages returned from append operation
        """

        geojson = dataframe.spatial.to_featureset().to_geojson
        result, messages = utils.retry(
            target_featurelayer.append,
            upload_format='geojson',
            edits=geojson,
            #:TODO figure out a way to preserve GUIDs?
            upsert=False,
            return_messages=True,
            rollback=True,
        )

        self._class_logger.debug(messages)
        if not result:
            raise RuntimeError(
                f'Failed to append data to layer id {layer_index} in itemid {feature_service_item_id}. Append should'
                ' have been rolled back.'
            )

        return messages

    def _save_truncated_data(self, dataframe, directory):
        """Save the pre-truncate dataframe to directory for safety

        Args:
            dataframe (pd.DataFrame): The data extracted from the feature layer prior to truncating
            directory (str or Path): The directory to save the data to.

        Returns:
            Path: The full path to the output file, named with today's date.
        """

        out_path = Path(directory, f'old_data_{datetime.date.today()}.json')
        out_path.write_text(dataframe.spatial.to_featureset().to_json)
        return out_path

    def truncate_and_load_feature_service(self, feature_service_item_id, new_dataframe, failsafe_dir, layer_index=0):
        """Attempt to delete existing data from a feature layer and add new data from a spatially-enabled dataframe.

        First attempts to truncate existing data. Then renames new data column names to conform to AGOL scheme (spaces,
        special chars changed to '_'). Finally attempts to append new data to now-empty feature layer. If the new data
        append fails, it attempts to re-upload the previous data from the in-memory dataframe. If this fails, it
        attempts to failsafe by writing the old data to disk as a json file.

        Args:
            feature_service_item_id (str): AGOL item ID for feature service to truncate and load
            new_dataframe (pd.DataFrame): Spatially-enabled dataframe containing new data. Must not contain columns
            that do not exist in the live data. Geometry type must match existing data.
            layer_index (int, optional): ID for the feature service layer to be truncated and loaded. Defaults to 0.

        Raises:
            RuntimeError: If new data contains a field not present in the live data
            RuntimeError: If the truncate fails
            RuntimeError: If append fails; will attempt to rollback appends that worked
            RuntimeError: If rollback fails; will attempt to write old live data to disk as json

        Returns:
            int: The number of new records added after deleting all existing records.
        """

        target_featurelayer = arcgis.features.FeatureLayer.fromitem(
            self.gis.content.get(feature_service_item_id), layer_id=layer_index
        )
        self._class_logger.info('Truncating existing data...')
        old_dataframe = self._truncate_existing_data(target_featurelayer, layer_index, feature_service_item_id)
        try:
            cleaned_dataframe = new_dataframe.rename(columns=utils.rename_columns_for_agol(new_dataframe.columns))
            #: temp fix until Esri fixes empty series as NaN bug
            fixed_dataframe = utils.replace_nan_series_with_empty_strings(cleaned_dataframe)
            utils.check_fields_match(target_featurelayer, fixed_dataframe)
            self._class_logger.info('Loading new data...')

            messages = self._append_new_data(target_featurelayer, fixed_dataframe, feature_service_item_id, layer_index)
        except Exception:
            try:
                self._class_logger.info('Append failed; attempting to re-load truncated data...')
                messages = self._append_new_data(
                    target_featurelayer, old_dataframe, feature_service_item_id, layer_index
                )
                self._class_logger.info('%s features reloaded', messages['recordCount'])
            except Exception as inner_error:
                failsafe_path = self._save_truncated_data(old_dataframe, failsafe_dir)
                raise RuntimeError(
                    f'Failed to re-add truncated data after failed append; data saved to {failsafe_path}'
                ) from inner_error
            finally:
                raise

        return messages['recordCount']


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
