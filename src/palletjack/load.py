"""Modify existing ArcGIS Online content (mostly hosted feature services). Contains classes for updating hosted feature
service data, modifying the attachments on a hosted feature service, or modifying map symbology.
"""

import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import arcgis
import numpy as np
import pandas as pd
from arcgis.features import GeoAccessor, GeoSeriesAccessor

from palletjack import utils

logger = logging.getLogger(__name__)


class FeatureServiceUpdater:
    """Update an AGOL Feature Service with data from a pandas DataFrame.

    Contains four class methods that can be called directly without needing to instantiate an object: add_features,
    remove_features, update_features, and truncate_and_load_features.

    It is the client's responsibility to separate out the new data into these different steps. If the extract/transform
    stages result in seperate groups of records that need to be added, deleted, and updated, the client must call the
    three different methods with dataframes containing only the respective records for each operation.

    Because the update process uploads the data as geojson, all input geometries must be in WGS84 (wkid 4326). Input
    dataframes can be projected using dataframe.spatial.project(4326). ArcGIS Online will then project the uploaded
    data to match the hosted feature service's projection.
    """

    @classmethod
    def add_features(cls, gis, feature_service_itemid, dataframe, layer_index=0):
        """Adds new features to existing hosted feature layer from new dataframe.

        The new dataframe must have a 'SHAPE' column containing geometries of the same type as the live data. The
        dataframe must have a WGS84 (wkid 4326) projection. New OBJECTIDs will be automatically generated.

        The new dataframe's columns and data must match the existing data's fields (with the exception of generated
        fields like shape area and length) in name, type, and allowable length. Live fields that are not nullable and
        don't have a default value must have a value in the new data; missing data in these fields will raise an error.

        Args:
            gis (arcgis.gis.GIS): GIS item for AGOL org
            features_service_item_id (str): itemid for service to update
            dataframe (pd.DataFrame.spatial): Spatially enabled dataframe of data to be added
            layer_index (int): Index of layer within service to update. Defaults to 0.

        Raises:
            ValueError: If the new field and existing fields don't match, the SHAPE field is missing or has an
                incompatible type, the new data contains null fields, the new data exceeds the existing field
                lengths, or a specified field is missing from either new or live data.

        Returns:
            int: Number of features added
        """
        updater = cls(gis, feature_service_itemid, dataframe, fields=list(dataframe.columns), layer_index=layer_index)
        return updater._add_new_data_to_hosted_feature_layer()

    @classmethod
    def remove_features(cls, gis, feature_service_itemid, delete_oids, layer_index=0):
        """Deletes features from a hosted feature layer based on comma-separated string of Object IDs

        This is a wrapper around the arcgis.FeatureLayer.delete_features method that adds some sanity checking. The
        delete operation is rolled-back if any of the features fail to delete using (rollback_on_failure=True). This
        function will raise a RuntimeError as well after delete_features() returns if any of them fail.

        The sanity checks will raise errors or warnings as appropriate if any of them fail.

        Args:
            delete_oids (list[int]): List of OIDs to delete

        Raises:
            ValueError: If delete_string can't be split on `,`
            TypeError: If any of the items in delete_string can't be cast to ints
            ValueError: If delete_string is empty
            UserWarning: If any of the Object IDs in delete_string don't exist in the live data
            RuntimeError: If any of the OIDs fail to delete

        Returns:
            int: The number of features deleted
        """

        updater = cls(gis, feature_service_itemid, layer_index=layer_index)
        return updater._delete_data_from_hosted_feature_layer(delete_oids)

    @classmethod
    def update_features(cls, gis, feature_service_itemid, dataframe, layer_index=0, update_geometry=True):
        """Updates existing features within a hosted feature layer using OBJECTID as the join field.

        The new dataframe's columns and data must match the existing data's fields (with the exception of generated
        fields like shape area and length) in name, type, and allowable length. Live fields that are not nullable and
        don't have a default value must have a value in the new data; missing data in these fields will raise an error.

        Uses the OBJECTID field to determine which features should be updated by the underlying FeatureLayer.append()
        method. The most robust way to do this is to load the live data as a dataframe, subset it down to the desired
        rows, make your edits based on a separate join id, and then pass that dataframe to this method.

        The new data can have either attributes and geometries or only attributes based on the update_geometry flag. A
        combination of updates from a source with both attributes & geometries and a source with attributes-only must
        be done with two separate calls. The geometries must be provided in a SHAPE column, be the same type as the
        live data, and have a WGS84 (wkid 4326) projection.

        Args:
            gis (arcgis.gis.GIS): GIS item for AGOL org
            features_service_item_id (str): itemid for service to update
            dataframe (pd.DataFrame.spatial): Spatially enabled dataframe of data to be updated
            layer_index (int): Index of layer within service to update. Defaults to 0.
            update_geometry (bool): Whether to update attributes and geometry (True) or just attributes (False).
                Defaults to False.

        Raises:
            ValueError: If the new field and existing fields don't match, the SHAPE field is missing or has an
                incompatible type, the new data contains null fields, the new data exceeds the existing field
                lengths, or a specified field is missing from either new or live data.

        Returns:
            int: Number of features updated
        """

        updater = cls(
            gis,
            feature_service_itemid,
            dataframe,
            fields=list(dataframe.columns),
            layer_index=layer_index,
        )
        return updater._update_hosted_feature_layer(update_geometry)

    @classmethod
    def truncate_and_load_features(cls, gis, feature_service_itemid, dataframe, failsafe_dir='', layer_index=0):
        """Overwrite a hosted feature layer by truncating and loading the new data

        When the existing dataset is truncated, a copy is kept in memory as a spatially-enabled dataframe. If the new
        data fail to load, this copy is reloaded. If the reload fails, the copy is written to failsafe_dir with the
        filename {todays_date}.json (2022-12-31.json).

        The new dataframe must have a 'SHAPE' column containing geometries of the same type as the live data. The
        dataframe must have a WGS84 (wkid 4326) projection. New OBJECTIDs will be automatically generated.

        The new dataframe's columns and data must match the existing data's fields (with the exception of generated
        fields like shape area and length) in name, type, and allowable length. Live fields that are not nullable and
        don't have a default value must have a value in the new data; missing data in these fields will raise an error.

        Args:
            gis (arcgis.gis.GIS): GIS item for AGOL org
            feature_service_itemid (str): itemid for service to update
            dataframe (pd.DataFrame.spatial): Spatially enabled dataframe of new data to be loaded
            failsafe_dir (str, optional): Directory to save original data in case of complete failure. If left blank,
            existing data won't be saved. Defaults to ''
            layer_index (int, optional): Index of layer within service to update. Defaults to 0.

        Returns:
            int: Number of features loaded
        """

        updater = cls(
            gis,
            feature_service_itemid,
            dataframe,
            fields=list(dataframe.columns),
            failsafe_dir=failsafe_dir,
            layer_index=layer_index
        )
        return updater._truncate_and_load_data()

    def __init__(self, gis, feature_service_itemid, dataframe=None, fields=None, failsafe_dir=None, layer_index=0):
        self._class_logger = logging.getLogger(__name__).getChild(self.__class__.__name__)
        self.feature_service_itemid = feature_service_itemid
        self.feature_layer = arcgis.features.FeatureLayer.fromitem(gis.content.get(feature_service_itemid))
        if dataframe is not None:
            self.new_dataframe = dataframe
            if 'SHAPE' in self.new_dataframe.columns:
                self.new_dataframe.spatial.set_geometry('SHAPE')
        if fields is not None:
            self.fields = list(set(fields) - {'Shape_Area', 'Shape_Length'})  #: We don't use these auto-gen fields
        self.failsafe_dir = failsafe_dir if failsafe_dir else None
        self.layer_index = layer_index

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

        #: Field checks to prevent various AGOL errors
        utils.FieldChecker.check_fields(self.feature_layer.properties, self.new_dataframe, self.fields, add_oid=False)

        #: Upsert
        append_count = self._upload_data(
            self.feature_layer,
            self.new_dataframe,
            upsert=False,
        )
        return append_count

    def _delete_data_from_hosted_feature_layer(self, delete_oids) -> int:
        """Deletes features from a hosted feature layer based on comma-separated string of OIDS

        Args:
            delete_oids (list[int]): List of OIDs to delete

        Raises:
            RuntimeError: If any of the OIDs fail to delete

        Returns:
            int: The number of features deleted
        """

        self._class_logger.info(
            'Deleting features from layer `%s` in itemid `%s`', self.layer_index, self.feature_service_itemid
        )
        self._class_logger.debug('Delete string: %s', delete_oids)

        #: Verify delete list
        # oid_list = utils.DeleteUtils.check_delete_oids_are_comma_separated(delete_oids)
        oid_numeric = utils.DeleteUtils.check_delete_oids_are_ints(delete_oids)
        utils.DeleteUtils.check_for_empty_oid_list(oid_numeric, delete_oids)
        delete_string = ','.join([str(oid) for oid in oid_numeric])
        num_missing_oids = utils.DeleteUtils.check_delete_oids_are_in_live_data(
            delete_string, oid_numeric, self.feature_layer
        )

        #: Note: apparently not all services support rollback: https://developers.arcgis.com/rest/services-reference/enterprise/delete-features.htm
        deletes = utils.retry(
            self.feature_layer.delete_features,
            deletes=delete_string,
            rollback_on_failure=True,
        )

        failed_deletes = [result['objectId'] for result in deletes['deleteResults'] if not result['success']]
        if failed_deletes:
            raise RuntimeError(f'The following Object IDs failed to delete: {failed_deletes}')

        #: The REST API still returns success: True on missing OIDs, so we have to track this ourselves
        actual_delete_count = len(deletes['deleteResults']) - num_missing_oids

        return actual_delete_count

    def _update_hosted_feature_layer(self, update_geometry) -> int:
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

        #: Add null geometries if update_geometry==False so that we can create a featureset from the dataframe
        #: (geometries will be ignored by upsert call)
        if not update_geometry:
            self._class_logger.debug('Attribute-only update; inserting null geometries')
            self.new_dataframe['SHAPE'] = utils.get_null_geometries(self.feature_layer.properties)

        #: Field checks to prevent various AGOL errors
        utils.FieldChecker.check_fields(self.feature_layer.properties, self.new_dataframe, self.fields, add_oid=True)

        #: Upsert
        append_count = self._upload_data(
            self.feature_layer,
            self.new_dataframe,
            upsert=True,
            upsert_matching_field='OBJECTID',
            append_fields=self.fields,  #: Apparently this works if append_fields is all the fields, but not a subset?
            update_geometry=update_geometry
        )
        return append_count

    #: TODO: rename this method? not everything is an upsert
    def _upload_data(self, target_featurelayer, dataframe, **append_kwargs):
        """UPdate and inSERT data into live dataset with featurelayer.append()

        Note: The call to to_featureset() in this method will add new OBJECTIDs to the new data if they aren't already present.

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

        #: FIXME: With chunking, rollback could leave the data in an inconsistent state if a chunk fails midway through the process. All the chunks before that won't be rolled back, so some of the data will be new and some will be old. I suppose this will be ok for upserts, where the data will just get updated again. However, adds will probably just create duplicates of the chunks that aren't rolled back if we try to re-do the whole add op again. Do we create a list of dataframe row indices that were in the successful chunks?

        running_append_total = 0
        geojsons = utils.Chunking.build_upload_json(dataframe, target_featurelayer.properties.fields, 20_000_000)
        self._class_logger.info('Appending/upserting data in %s chunk(s)', len(geojsons))
        chunk_sizes = []
        for chunk, geojson in enumerate(geojsons, 1):
            try:
                chunk_start = datetime.now()
                chunk_size = sys.getsizeof(geojson.encode('utf-16'))
                chunk_sizes.append(chunk_size)
                self._class_logger.debug('Uploading chunk %s of %s, %s bytes', chunk, len(geojsons), chunk_size)
                result, messages = utils.retry(
                    target_featurelayer.append,
                    upload_format='geojson',
                    edits=geojson,
                    return_messages=True,
                    rollback=True,
                    **append_kwargs
                )
                self._class_logger.debug(messages)
                self._class_logger.debug('Chunk time: %s', datetime.now() - chunk_start)
            except Exception:
                self._class_logger.debug(pd.Series(chunk_sizes).describe())
                self._class_logger.error('Append failed, feature service may be dirty due to append chunking.')
                raise
            if not result:
                self._class_logger.debug(pd.Series(chunk_sizes).describe())
                raise RuntimeError(
                    f'Failed to append data at chunk {chunk} of {len(geojsons)}. Append operation should have been rolled back.'
                )

            running_append_total += messages['recordCount']

        self._class_logger.debug(pd.Series(chunk_sizes).describe())
        return running_append_total

    def _truncate_and_load_data(self):
        """Overwrite a layer by truncating and loading new data

        Raises:
            RuntimeError: If loading fails and reloading of old data fails (old data will be written to disk)

        Returns:
            int: Number of features loaded
        """

        self._class_logger.info(
            'Truncating and loading layer `%s` in itemid `%s`', self.layer_index, self.feature_service_itemid
        )
        start = datetime.now()

        #: Save the data to disk if failsafe dir provided
        #: TODO: return path programmatically so client can catch exception and try to reload automatically?
        if self.failsafe_dir:
            self._class_logger.info('Saving existing data to %s', self.failsafe_dir)
            saved_layer_path = utils.save_feature_layer_to_json(self.feature_layer, self.failsafe_dir)

        #: Field checks to prevent various AGOL errors
        utils.FieldChecker.check_fields(self.feature_layer.properties, self.new_dataframe, self.fields, add_oid=False)

        self._class_logger.info('Truncating existing features...')
        self._truncate_existing_data()

        try:
            self._class_logger.info('Loading new data...')
            append_count = self._upload_data(self.feature_layer, self.new_dataframe, upsert=False)
            self._class_logger.debug('Total truncate and load time: %s', datetime.now() - start)
        except Exception:
            if self.failsafe_dir:
                self._class_logger.error(
                    'Append failed, feature service may be dirty due to append chunking. Data saved to %s',
                    saved_layer_path
                )
                raise
            self._class_logger.error(
                'Append failed, feature service may be dirty due to append chunking. Old data not saved (no failsafe dir set)'
            )
            raise

        return append_count

    def _truncate_existing_data(self):
        """Remove all existing features from the live dataset

        Raises:
            RuntimeError: If the truncate fails

        Returns:
            pd.DataFrame: The feature layer's data as a spatially-enabled dataframe prior to truncating
        """

        self._class_logger.debug('Truncating...')
        truncate_result = utils.retry(self.feature_layer.manager.truncate, asynchronous=True, wait=True)
        self._class_logger.debug(truncate_result)
        if truncate_result['status'] != 'Completed':
            raise RuntimeError(
                f'Failed to truncate existing data from layer id {self.layer_index} in itemid {self.feature_service_itemid}'
            )


class FeatureServiceAttachmentsUpdater:
    """Add or overwrite attachments in a feature service using a dataframe of the desired "new" attachments.

    Updates the attachments based on a dataframe containing two columns: a join key present in the live data (the
    dataframe column name must match the feature service field name) and the path of the file to attach to the feature.
    While AGOL supports multiple attachments, this only accepts a single file as input.

    If a matching feature in AGOl doesn't have an attachment, the file referred to by the dataframe will be uploaded.
    If it does have an attachment, it checks the existing filename with the referenced file. If they are different, the
    file from the dataframe will be updated. If they are the same, nothing happens.
    """

    def __init__(self, gis):
        """
        Args:
            gis (arcgis.gis.GIS): The AGOL organization's gis object
        """

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


class ColorRampReclassifier:
    """Updates the interval ranges on a webmap's layer's classification renderer based on the layer's current data.

    Manually edits the JSON definition to change a layer's color ramp values based on a simple unclassed scheme similar
    to AGOL's unclassed ramp. The minimum value is the dataset minimum, the max is the mean value plus one standard
    deviation.
    """

    def __init__(self, webmap_item, gis):
        """
        Args:
            webmap_item (arcgis.mapping.WebMap): The webmap item in the AGOL organization
            gis (arcgis.gis.GIS): The AGOL organization as a gis object
        """

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
