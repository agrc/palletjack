"""Modify existing ArcGIS Online content (mostly hosted feature services). Contains classes for updating hosted feature
service data, modifying the attachments on a hosted feature service, or modifying map symbology.
"""

import json
import logging
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import arcgis
import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from arcgis.features import GeoAccessor, GeoSeriesAccessor

from palletjack import utils

logger = logging.getLogger(__name__)


class FeatureServiceUpdater:
    """Update an AGOL Feature Service with data from a pandas DataFrame.

    This class represents the feature layer that will be updated and stores a reference to the feature layer and it's
    containing gis. It contains four methods for updating the layer's data: add_features, remove_features,
    update_features, and truncate_and_load_features.

    It is the client's responsibility to separate out the new data into these different steps. If the extract/transform
    stages result in separate groups of records that need to be added, deleted, and updated, the client must call the
    three different methods with dataframes containing only the respective records for each operation.

    The method used to upload the data to AGOL saves the updated data as a new layer named upload in working_dir/upload.
    gdb, zips the gdb, uploads it to AGOL, and then uses this as the source data for a call to the feature layer's
    .append() method. The geodatabase upload.gdb will be created in working_dir if it doesn't already exist. Ideally,
    working_dir should be a TemporaryDirectory unless persistent access to the gdb is desired.
    """

    def __init__(self, gis, feature_service_itemid, working_dir=None, layer_index=0):
        self._class_logger = logging.getLogger(__name__).getChild(self.__class__.__name__)
        self.gis = gis
        self.feature_service_itemid = feature_service_itemid
        self.feature_layer = arcgis.features.FeatureLayer.fromitem(gis.content.get(feature_service_itemid))
        self.working_dir = working_dir if working_dir else None
        self.layer_index = layer_index

    def add_features(self, dataframe):
        """Adds new features to existing hosted feature layer from new dataframe.

        The new dataframe must have a 'SHAPE' column containing geometries of the same type as the live data.

        The new dataframe's columns and data must match the existing data's fields (with the exception of generated
        fields like shape area and length) in name, type, and allowable length. Live fields that are not nullable and
        don't have a default value must have a value in the new data; missing data in these fields will raise an error.

        Args:
            dataframe (pd.DataFrame.spatial): Spatially enabled dataframe of data to be added

        Raises:
            ValueError: If the new field and existing fields don't match, the SHAPE field is missing or has an
                incompatible type, the new data contains null fields, the new data exceeds the existing field
                lengths, or a specified field is missing from either new or live data.

        Returns:
            int: Number of features added
        """

        self._class_logger.info(
            'Adding items to layer `%s` in itemid `%s` in-place', self.layer_index, self.feature_service_itemid
        )
        fields = FeatureServiceUpdater._get_fields_from_dataframe(dataframe)
        self._class_logger.debug('Using fields %s', fields)

        #: Field checks to prevent various AGOL errors
        utils.FieldChecker.check_fields(self.feature_layer.properties, dataframe, fields, add_oid=False)

        #: Upload
        append_count = self._upload_data(
            dataframe,
            upsert=False,
        )
        return append_count

    def remove_features(self, delete_oids):
        """Deletes features from a hosted feature layer based on comma-separated string of Object IDs

        This is a wrapper around the arcgis.FeatureLayer.delete_features method that adds some sanity checking. The
        delete operation is rolled back if any of the features fail to delete using (rollback_on_failure=True). This
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

        self._class_logger.info(
            'Deleting features from layer `%s` in itemid `%s`', self.layer_index, self.feature_service_itemid
        )
        self._class_logger.debug('Delete string: %s', delete_oids)

        #: Verify delete list
        oid_numeric = utils.DeleteUtils.check_delete_oids_are_ints(delete_oids)
        utils.DeleteUtils.check_for_empty_oid_list(oid_numeric, delete_oids)
        delete_string = ','.join([str(oid) for oid in oid_numeric])
        num_missing_oids = utils.DeleteUtils.check_delete_oids_are_in_live_data(
            delete_string, oid_numeric, self.feature_layer
        )

        #: Note: apparently not all services support rollback:
        #: https://developers.arcgis.com/rest/services-reference/enterprise/delete-features.htm
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

    def update_features(self, dataframe, update_geometry=True):
        """Updates existing features within a hosted feature layer using OBJECTID as the join field.

        The new dataframe's columns and data must match the existing data's fields (with the exception of generated
        fields like shape area and length) in name, type, and allowable length. Live fields that are not nullable and
        don't have a default value must have a value in the new data; missing data in these fields will raise an error.

        Uses the OBJECTID field to determine which features should be updated by the underlying FeatureLayer.append()
        method. The most robust way to do this is to load the live data as a dataframe, subset it down to the desired
        rows, make your edits based on a separate join id, and then pass that dataframe to this method.

        The new data can have either attributes and geometries or only attributes based on the update_geometry flag. A
        combination of updates from a source with both attributes & geometries and a source with attributes-only must
        be done with two separate calls. The geometries must be provided in a SHAPE column and be the same type as the
        live data.

        Args:
            dataframe (pd.DataFrame.spatial): Spatially enabled dataframe of data to be updated
            update_geometry (bool): Whether to update attributes and geometry (True) or just attributes (False).
                Defaults to False.

        Raises:
            ValueError: If the new field and existing fields don't match, the SHAPE field is missing or has an
                incompatible type, the new data contains null fields, the new data exceeds the existing field
                lengths, or a specified field is missing from either new or live data.

        Returns:
            int: Number of features updated
        """

        self._class_logger.info(
            'Updating layer `%s` in itemid `%s` in-place', self.layer_index, self.feature_service_itemid
        )

        fields = FeatureServiceUpdater._get_fields_from_dataframe(dataframe)
        self._class_logger.debug('Updating fields %s', fields)

        #: Add null geometries if update_geometry==False so that we can create a featureset from the dataframe
        #: (geometries will be ignored by upsert call)
        if not update_geometry:
            self._class_logger.debug('Attribute-only update; inserting null geometries')
            dataframe['SHAPE'] = utils.get_null_geometries(self.feature_layer.properties)

        #: Field checks to prevent various AGOL errors
        utils.FieldChecker.check_fields(self.feature_layer.properties, dataframe, fields, add_oid=True)

        #: Upload data
        append_count = self._upload_data(
            dataframe,
            upsert=True,
            upsert_matching_field='OBJECTID',
            append_fields=fields,  #: Apparently this works if append_fields is all the fields, but not a subset?
            update_geometry=update_geometry
        )
        return append_count

    def truncate_and_load_features(self, dataframe, save_old=False):
        """Overwrite a hosted feature layer by truncating and loading the new data

        When the existing dataset is truncated, a copy is kept in memory as a spatially-enabled dataframe. If
        save_old is set, this is saved as a layer in self.working_dir/backup.gdb with the layer name
        {featurelayer.name}_{todays_date}.json (foobar_2022-12-31.json).

        The new dataframe must have a 'SHAPE' column containing geometries of the same type as the live data. New
        OBJECTIDs will be automatically generated.

        The new dataframe's columns and data must match the existing data's fields (with the exception of generated
        fields like shape area and length) in name, type, and allowable length. Live fields that are not nullable and
        don't have a default value must have a value in the new data; missing data in these fields will raise an error.

        Args:
            dataframe (pd.DataFrame.spatial): Spatially enabled dataframe of new data to be loaded
            save_old (bool, optional): Save existing data to backup.gdb in working_dir. Defaults to False

        Returns:
            int: Number of features loaded
        """

        self._class_logger.info(
            'Truncating and loading layer `%s` in itemid `%s`', self.layer_index, self.feature_service_itemid
        )
        start = datetime.now()

        #: Save the data to disk if desired
        if save_old:
            self._class_logger.info('Saving existing data to %s', self.working_dir)
            saved_layer_path = utils.save_feature_layer_to_gdb(self.feature_layer, self.working_dir)

        fields = FeatureServiceUpdater._get_fields_from_dataframe(dataframe)

        #: Field checks to prevent various AGOL errors
        utils.FieldChecker.check_fields(self.feature_layer.properties, dataframe, fields, add_oid=False)

        self._class_logger.info('Truncating existing features...')
        self._truncate_existing_data()

        try:
            self._class_logger.info('Loading new data...')
            append_count = self._upload_data(dataframe, upsert=False)
            self._class_logger.debug('Total truncate and load time: %s', datetime.now() - start)
        except Exception:
            if save_old:
                self._class_logger.error('Append failed. Data saved to %s', saved_layer_path)
                raise
            self._class_logger.error('Append failed. Old data not saved (save_old set to False)')
            raise

        return append_count

    @staticmethod
    def _get_fields_from_dataframe(dataframe):
        """Get the fields from a dataframe, excluding Shape_Area and Shape_Length

        Args:
            dataframe (pd.DataFrame): Dataframe to get fields from

        Returns:
            list[str]: List of the columns of the dataframe, excluding Shape_Area and Shape_Length
        """

        fields = list(dataframe.columns)
        for auto_gen_field in ['Shape_Area', 'Shape_Length']:
            try:
                fields.remove(auto_gen_field)
            except ValueError:
                continue

        return fields

    def _upload_data(self, dataframe, **append_kwargs):
        """Append a spatially-enabled dataframe to a feature layer by uploading it as a zipped file gdb

        We first save the new dataframe as a layer in an empty geodatabase, then zip it and upload it to AGOL as a
        standalone item. We then call append on the target feature layer with this item as the source for the append,
        using upsert where appropriate to update existing data using OBJECTID as the join field. Afterwards, we delete
        the gdb item and the zipped gdb.

        Args:
            dataframe (pd.DataFrame.spatial): A spatially-enabled dataframe containing data to be added or upserted to
            the feature layer. The fields must match the live fields in name, type, and length (where applicable). The
            dataframe must have a SHAPE column containing geometries of the same type as the live data.

        Raises:
            ValueError: If the field used as a key for upsert matching is not present in either the new or live data
            RuntimeError: If the append operation fails

        Returns:
            int: The number of records upserted
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

        zipped_gdb_path = self._save_to_gdb_and_zip(dataframe)

        gdb_item = self._upload_gdb(zipped_gdb_path)

        try:
            result, messages = utils.retry(
                self.feature_layer.append,
                item_id=gdb_item.id,
                upload_format='filegdb',
                source_table_name='upload',
                return_messages=True,
                rollback=True,
                **append_kwargs
            )
            if not result:
                raise RuntimeError('Append failed but did not error')
        except Exception as error:
            raise RuntimeError('Failed to append data from gdb, changes should have been rolled back') from error

        self._cleanup(gdb_item, zipped_gdb_path)

        return messages['recordCount']

    def _save_to_gdb_and_zip(self, dataframe):
        """Save a spatially-enabled dataframe to a gdb, zip it, and return path to the zipped file.

        Requires self.working_dir to be set. Uses pyogrio to save the dataframe to the gdb, then uses
        shutil.make_archive to zip the gdb. The zipped gdb is saved in self.working_dir.

        Args:
            dataframe (pd.DataFrame.spatial): The input dataframe to be saved. Must contain geometries in the SHAPE
            field

        Raises:
            ValueError: If self.working_dir is not set or the empty upload.gdb doesn't exist in it

        Returns:
            pathlib.Path: The path to the zipped GDB
        """

        try:
            gdb_path = Path(self.working_dir) / 'upload.gdb'
        except TypeError as error:
            raise AttributeError('working_dir not specified on FeatureServiceUpdater') from error

        gdf = utils.sedf_to_gdf(dataframe)

        try:
            gdf.to_file(gdb_path, layer='upload', engine='pyogrio', driver='OpenFileGDB')
        except pyogrio.errors.DataSourceError as error:
            raise ValueError(
                f'Error writing layer to {gdb_path}. Verify {self.working_dir} exists and is writable.'
            ) from error
        try:
            zipped_gdb_path = shutil.make_archive(gdb_path, 'zip', root_dir=gdb_path.parent, base_dir=gdb_path.name)
        except OSError as error:
            raise ValueError(f'Error zipping {gdb_path}') from error

        return zipped_gdb_path

    def _upload_gdb(self, zipped_gdb_path):
        """Add a zipped gdb to AGOL as an item to self.gis

        Args:
            zipped_gdb_path (str or Path-like): Path to the zipped gdb

        Raises:
            RuntimeError: If there is an error uploading the gdb to AGOL

        Returns:
            arcgis.gis.Item: Reference to the resulting Item object in self.gis
        """

        try:
            gdb_item = utils.retry(
                self.gis.content.add,
                item_properties={
                    'type': 'File Geodatabase',
                    'title': 'Temporary gdb upload',
                    'snippet': 'Temporary gdb upload from palletjack'
                },
                data=zipped_gdb_path
            )
        except Exception as error:
            raise RuntimeError(f'Error uploading {zipped_gdb_path} to AGOL') from error
        return gdb_item

    def _cleanup(self, gdb_item, zipped_gdb_path):
        """Remove the zipped gdb from disk and the gdb item from AGOL

        Args:
            gdb_item (arcgis.gis.Item): Reference to the gdb item in self.gis
            zipped_gdb_path (str or Path-like): Path to the gdb on disk

        Raises:
            RuntimeError: If there are errors deleting the gdb item or the zipped gdb
        """

        try:
            gdb_item.delete()
        except Exception as error:
            warnings.warn(f'Error deleting gdb item {gdb_item.id} from AGOL')
            warnings.warn(repr(error))

        try:
            Path(zipped_gdb_path).unlink()
        except Exception as error:
            warnings.warn(f'Error deleting zipped gdb {zipped_gdb_path}')
            warnings.warn(repr(error))

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
