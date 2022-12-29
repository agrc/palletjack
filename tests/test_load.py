import json
import logging
import re
import sys
import urllib
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest
from arcgis import GeoAccessor, GeoSeriesAccessor

from palletjack import load


class TestFeatureServiceUpdaterInit:

    def test_init_calls_get_feature_layer(self, mocker):
        new_dataframe = pd.DataFrame(columns=['Foo field', 'Bar'])
        arcgis_mock = mocker.patch('palletjack.load.arcgis')

        updater = load.FeatureServiceUpdater(mocker.Mock(), 'itemid', new_dataframe)

        arcgis_mock.features.FeatureLayer.fromitem.assert_called_once()

    def test_init_renames_dataframe_columns(self, mocker):

        new_dataframe = pd.DataFrame(columns=['Foo field', 'Bar'])
        mocker.patch('palletjack.load.arcgis')

        updater = load.FeatureServiceUpdater(mocker.Mock(), 'itemid', new_dataframe, join_column='Bar')

        assert list(updater.new_dataframe.columns) == ['Foo_field', 'Bar']

    def test_init_renames_fields(self, mocker):

        new_dataframe = pd.DataFrame(columns=['Foo', 'Bar'])
        mocker.patch('palletjack.load.arcgis')
        fields = ['Foo field', 'Bar!field', 'baz']

        updater = load.FeatureServiceUpdater(mocker.Mock(), 'itemid', new_dataframe, fields=fields)

        assert set(updater.fields) == {'Foo_field', 'Bar_field', 'baz'}

    def test_init_renames_join_column(self, mocker):

        new_dataframe = pd.DataFrame(columns=['Foo', 'Bar'])
        mocker.patch('palletjack.load.arcgis')

        updater = load.FeatureServiceUpdater(mocker.Mock(), 'itemid', new_dataframe, join_column='Foo field')

        assert updater.join_column == 'Foo_field'


class TestUpdateLayer:

    def test_update_hosted_feature_layer_calls_upsert_correctly(self, mocker):
        new_dataframe = pd.DataFrame({
            'foo': [1, 2],
            'bar': [3, 4],
            'OBJECTID': [11, 12],
            'SHAPE': ['s1', 's2'],
        })
        updater_mock = mocker.Mock()
        updater_mock.feature_service_itemid = 'foo123'
        fl_mock = mocker.Mock()
        updater_mock.feature_layer = fl_mock
        updater_mock.new_dataframe = new_dataframe
        updater_mock.fields = list(new_dataframe.columns)
        updater_mock.join_column = 'OBJECTID'
        updater_mock.layer_index = 0

        updater_mock._upsert_data.return_value = {'recordCount': 1}

        field_checker_mock = mocker.patch('palletjack.utils.FieldChecker')

        load.FeatureServiceUpdater._update_hosted_feature_layer(updater_mock, update_geometry=True)

        updater_mock._upsert_data.assert_called_once_with(
            fl_mock,
            new_dataframe,
            upsert=True,
            upsert_matching_field='OBJECTID',
            append_fields=['foo', 'bar', 'OBJECTID', 'SHAPE'],
            update_geometry=True
        )

    def test_update_hosted_feature_layer_calls_field_checkers(self, mocker):
        new_dataframe = pd.DataFrame({
            'foo': [1, 2],
            'bar': [3, 4],
            'OBJECTID': [11, 12],
        })
        updater_mock = mocker.Mock()
        updater_mock.feature_service_itemid = 'foo123'
        updater_mock.feature_layer = mocker.Mock()
        updater_mock.new_dataframe = new_dataframe
        updater_mock.fields = list(new_dataframe.columns)
        updater_mock.join_column = 'OBJECTID'
        updater_mock.layer_index = 0

        updater_mock._upsert_data.return_value = {'recordCount': 1}

        field_checker_mock = mocker.patch('palletjack.utils.FieldChecker')

        load.FeatureServiceUpdater._update_hosted_feature_layer(updater_mock, update_geometry=True)

        field_checker_mock.return_value.check_live_and_new_field_types_match.assert_called_once_with([
            'foo', 'bar', 'OBJECTID'
        ])
        field_checker_mock.return_value.check_for_non_null_fields.assert_called_once_with(['foo', 'bar', 'OBJECTID'])
        field_checker_mock.return_value.check_field_length.assert_called_once_with(['foo', 'bar', 'OBJECTID'])
        field_checker_mock.return_value.check_fields_present.assert_called_once_with(['foo', 'bar', 'OBJECTID'],
                                                                                     add_oid=True)

    def test_update_hosted_feature_layer_no_geometry_calls_null_geometry_generator(self, mocker):
        new_dataframe = pd.DataFrame({
            'foo': [1, 2],
            'bar': [3, 4],
            'OBJECTID': [11, 12],
        })
        updater_mock = mocker.Mock()
        updater_mock.feature_service_itemid = 'foo123'
        updater_mock.feature_layer = mocker.Mock()
        updater_mock.feature_layer.properties = {'geometryType': 'esriGeometryPoint'}
        updater_mock.new_dataframe = new_dataframe
        # updater_mock.fields = ['foo', 'bar']
        updater_mock.join_column = 'OBJECTID'
        updater_mock.layer_index = 0

        updater_mock._upsert_data.return_value = {'recordCount': 1}

        field_checker_mock = mocker.patch('palletjack.utils.FieldChecker')
        null_generator_mock = mocker.patch('palletjack.utils.get_null_geometries', return_value='Nullo')

        load.FeatureServiceUpdater._update_hosted_feature_layer(updater_mock, update_geometry=False)

        null_generator_mock.assert_called_once_with({'geometryType': 'esriGeometryPoint'})

    def test_update_hosted_feature_layer_no_geometry_calls_upsert_correctly(self, mocker):
        new_dataframe = pd.DataFrame({
            'foo': [1, 2],
            'bar': [3, 4],
            'OBJECTID': [11, 12],
        })
        updater_mock = mocker.Mock()
        updater_mock.feature_service_itemid = 'foo123'
        updater_mock.feature_layer = mocker.Mock()
        updater_mock.feature_layer.properties = {'geometryType': 'esriGeometryPoint'}
        updater_mock.new_dataframe = new_dataframe
        updater_mock.fields = list(new_dataframe.columns)
        updater_mock.join_column = 'OBJECTID'
        updater_mock.layer_index = 0

        updater_mock._upsert_data.return_value = {'recordCount': 1}

        field_checker_mock = mocker.patch('palletjack.utils.FieldChecker')
        null_generator_mock = mocker.patch('palletjack.utils.get_null_geometries', return_value='Nullo')

        load.FeatureServiceUpdater._update_hosted_feature_layer(updater_mock, update_geometry=False)

        updater_mock._upsert_data.assert_called_once_with(
            updater_mock.feature_layer,
            new_dataframe,
            upsert=True,
            upsert_matching_field='OBJECTID',
            append_fields=['foo', 'bar', 'OBJECTID'],
            update_geometry=False
        )


class TestAddToLayer:

    def test_add_new_data_to_hosted_feature_layer_calls_upsert(self, mocker):
        new_dataframe = pd.DataFrame({
            'foo': [1, 2],
            'bar': [3, 4],
            # 'OBJECTID': [11, 12],
        })
        updater_mock = mocker.Mock()
        updater_mock.feature_service_itemid = 'foo123'
        updater_mock.feature_layer = mocker.Mock()
        updater_mock.new_dataframe = new_dataframe
        updater_mock.fields = ['foo', 'bar']
        # updater_mock.join_column = 'OBJECTID'
        updater_mock.layer_index = 0

        updater_mock._upsert_data.return_value = {'recordCount': 1}

        field_checker_mock = mocker.patch('palletjack.utils.FieldChecker')

        load.FeatureServiceUpdater._add_new_data_to_hosted_feature_layer(updater_mock)

        updater_mock._upsert_data.assert_called_once_with(
            updater_mock.feature_layer,
            new_dataframe,
            upsert=False,
        )

    def test_add_new_data_to_hosted_feature_layer_calls_field_checkers(self, mocker):
        new_dataframe = pd.DataFrame({
            'foo': [1, 2],
            'bar': [3, 4],
            # 'OBJECTID': [11, 12],
        })
        updater_mock = mocker.Mock()
        updater_mock.feature_service_itemid = 'foo123'
        updater_mock.feature_layer = mocker.Mock()
        updater_mock.new_dataframe = new_dataframe
        updater_mock.fields = list(new_dataframe.columns)
        # updater_mock.join_column = 'OBJECTID'
        updater_mock.layer_index = 0

        updater_mock._upsert_data.return_value = {'recordCount': 1}

        field_checker_mock = mocker.patch('palletjack.utils.FieldChecker')

        load.FeatureServiceUpdater._add_new_data_to_hosted_feature_layer(updater_mock)

        field_checker_mock.return_value.check_live_and_new_field_types_match.assert_called_once_with(['foo', 'bar'])
        field_checker_mock.return_value.check_for_non_null_fields.assert_called_once_with(['foo', 'bar'])
        field_checker_mock.return_value.check_field_length.assert_called_once_with(['foo', 'bar'])
        field_checker_mock.return_value.check_fields_present.assert_called_once_with(['foo', 'bar'], add_oid=False)


class TestDeleteFromLayer:

    def test_delete_data_from_hosted_feature_layer_returns_number_of_deleted_features(self, mocker):
        updater_mock = mocker.Mock()
        updater_mock.feature_service_itemid = 'foo123'
        updater_mock.layer_index = 0
        updater_mock.feature_layer.delete_features.return_value = {
            'deleteResults': [
                {'objectId': 11, 'uniqueId': 11, 'globalId': None, 'success': True},
                {'objectId': 17, 'uniqueId': 17, 'globalId': None, 'success': True},
            ]
        }  # yapf: disable

        delete_utils_mock = mocker.patch('palletjack.utils.DeleteUtils')
        delete_utils_mock.check_delete_oids_are_in_live_data.return_value = 0

        deleted = load.FeatureServiceUpdater._delete_data_from_hosted_feature_layer(updater_mock, '11,17')

        assert deleted == 2

    def test_delete_data_from_hosted_feature_layer_raises_on_failed_delete(self, mocker):
        updater_mock = mocker.Mock()
        updater_mock.feature_service_itemid = 'foo123'
        updater_mock.layer_index = 0
        updater_mock.feature_layer.delete_features.return_value = {
            'deleteResults': [
                {'objectId': 11, 'uniqueId': 11, 'globalId': None, 'success': True},
                {'objectId': 17, 'uniqueId': 17, 'globalId': None, 'success': False},
            ]
        }  # yapf: disable

        mocker.patch('palletjack.utils.DeleteUtils')

        with pytest.raises(RuntimeError, match='The following Object IDs failed to delete: \[17\]'):  # as exc_info:
            deleted = load.FeatureServiceUpdater._delete_data_from_hosted_feature_layer(updater_mock, '11,17')

        # assert  exc_info.value.args[0] == 'The following Object IDs failed to delete: 17' in str(exc_info.value)

    def test_delete_data_from_hosted_feature_layer_runs_proper_check_sequence(self, mocker):
        updater_mock = mocker.Mock()
        updater_mock.feature_service_itemid = 'foo123'
        updater_mock.layer_index = 0
        updater_mock.feature_layer.query.return_value = {'objectIdFieldName': 'OBJECTID', 'objectIds': [11, 17]}
        updater_mock.feature_layer.delete_features.return_value = {
            'deleteResults': [
                {'objectId': 11, 'uniqueId': 11, 'globalId': None, 'success': True},
            ]
        }  # yapf: disable

        deleted = load.FeatureServiceUpdater._delete_data_from_hosted_feature_layer(updater_mock, '11')

        assert deleted == 1


class TestTruncateAndLoadLayer:

    def test_truncate_existing_data_normal(self, mocker):
        updater_mock = mocker.Mock()
        updater_mock.feature_service_itemid = 'foo123'
        updater_mock.layer_index = 0

        updater_mock.feature_layer.manager.truncate.return_value = {
            'submissionTime': 123,
            'lastUpdatedTime': 124,
            'status': 'Completed',
        }

        load.FeatureServiceUpdater._truncate_existing_data(updater_mock)

    def test_truncate_existing_raises_error_on_failure(self, mocker):
        updater_mock = mocker.Mock()
        updater_mock.feature_service_itemid = 'foo123'
        updater_mock.layer_index = 0
        updater_mock.feature_layer.manager.truncate.return_value = {
            'submissionTime': 123,
            'lastUpdatedTime': 124,
            'status': 'Foo',
        }

        with pytest.raises(RuntimeError, match='Failed to truncate existing data from layer id 0 in itemid foo123'):
            load.FeatureServiceUpdater._truncate_existing_data(updater_mock)

    def test_truncate_existing_retries_on_HTTPError(self, mocker):
        updater_mock = mocker.Mock()
        updater_mock.feature_service_itemid = 'foo123'
        updater_mock.layer_index = 0
        updater_mock.feature_layer.manager.truncate.side_effect = [
            urllib.error.HTTPError('url', 'code', 'msg', 'hdrs', 'fp'), {
                'submissionTime': 123,
                'lastUpdatedTime': 124,
                'status': 'Completed',
            }
        ]

        load.FeatureServiceUpdater._truncate_existing_data(updater_mock)

    def test_truncate_and_load_feature_service_normal(self, mocker):
        updater_mock = mocker.Mock()
        updater_mock.feature_service_itemid = 'foo123'
        updater_mock.layer_index = 0

        updater_mock.new_dataframe = pd.DataFrame(columns=['Foo', 'Bar'])

        mocker.patch('palletjack.utils.FieldChecker')

        updater_mock._upsert_data.return_value = {'recordCount': 42}

        uploaded_features = load.FeatureServiceUpdater._truncate_and_load_data(updater_mock)

        assert uploaded_features == 42

    def test_truncate_and_load_append_fails_reload_works(self, mocker, caplog):
        caplog.set_level(logging.DEBUG)

        updater_mock = mocker.Mock()
        updater_mock._class_logger = logging.getLogger('mock logger')
        updater_mock.feature_service_itemid = 'foo123'
        updater_mock.layer_index = 0
        updater_mock.new_dataframe = pd.DataFrame(columns=['Foo', 'Bar'])
        updater_mock._truncate_existing_data.return_value = 'old_data'

        # mocker.patch('palletjack.utils.replace_nan_series_with_bogus_value', return_value='new_data')
        mocker.patch('palletjack.utils.FieldChecker')
        mocker.patch('palletjack.utils.sleep')

        updater_mock._upsert_data.side_effect = [
            RuntimeError('Failed to append data. Append operation should have been rolled back.'), {
                'recordCount': 42
            }
        ]

        with pytest.raises(RuntimeError, match='Failed to append data. Append operation should have been rolled back.'):
            uploaded_features = load.FeatureServiceUpdater._truncate_and_load_data(updater_mock)

        assert updater_mock._upsert_data.call_args_list[0].args == (
            updater_mock.feature_layer, updater_mock.new_dataframe
        )
        assert updater_mock._upsert_data.call_args_list[0].kwargs == {'upsert': False}

        assert updater_mock._upsert_data.call_args_list[1].args == (updater_mock.feature_layer, 'old_data')
        assert updater_mock._upsert_data.call_args_list[1].kwargs == {'upsert': False}

        assert 'Append failed; attempting to re-load truncated data...' in caplog.text
        assert '42 features reloaded' in caplog.text

    def test_truncate_and_load_calls_field_checkers(self, mocker, caplog):

        caplog.set_level(logging.DEBUG)

        updater_mock = mocker.Mock()
        updater_mock._class_logger = logging.getLogger('mock logger')
        updater_mock.feature_service_itemid = 'foo123'
        updater_mock.layer_index = 0
        updater_mock.fields = ['Foo', 'Bar']
        updater_mock._truncate_existing_data.return_value = 'old_data'

        mocker.patch('palletjack.utils.sleep')
        field_checker_mock = mocker.patch('palletjack.utils.FieldChecker')

        updater_mock._upsert_data.return_value = {'recordCount': 42}

        uploaded_features = load.FeatureServiceUpdater._truncate_and_load_data(updater_mock)

        field_checker_mock.return_value.check_live_and_new_field_types_match.assert_called_once_with(['Foo', 'Bar'])
        field_checker_mock.return_value.check_for_non_null_fields.assert_called_once_with(['Foo', 'Bar'])
        field_checker_mock.return_value.check_field_length.assert_called_once_with(['Foo', 'Bar'])
        field_checker_mock.return_value.check_fields_present.assert_called_once_with(['Foo', 'Bar'], add_oid=False)
        assert uploaded_features == 42

    def test_truncate_and_load_saves_to_json_after_append_and_reload_fail(self, mocker, caplog):
        caplog.set_level(logging.DEBUG)

        updater_mock = mocker.Mock()
        updater_mock._class_logger = logging.getLogger('mock logger')
        updater_mock.feature_service_itemid = 'foo123'
        updater_mock.layer_index = 0
        updater_mock.new_dataframe = pd.DataFrame(columns=['Foo', 'Bar'])
        updater_mock.failsafe_dir = '/foo'
        updater_mock._truncate_existing_data.return_value = 'old_data'

        mocker.patch('palletjack.utils.FieldChecker')
        save_mock = mocker.patch(
            'palletjack.utils.save_spatially_enabled_dataframe_to_json', return_value='/foo/bar.json'
        )
        mocker.patch('palletjack.utils.sleep')

        updater_mock._upsert_data.side_effect = [
            RuntimeError('Failed to append data. Append operation should have been rolled back.'),
            RuntimeError('Failed to append data. Append operation should have been rolled back.')
        ]

        with pytest.raises(
            RuntimeError,
            match=re.escape('Failed to re-add truncated data after failed append; data saved to /foo/bar.json')
        ):
            uploaded_features = load.FeatureServiceUpdater._truncate_and_load_data(updater_mock)

        assert updater_mock._upsert_data.call_args_list[0].args == (
            updater_mock.feature_layer, updater_mock.new_dataframe
        )
        assert updater_mock._upsert_data.call_args_list[0].kwargs == {'upsert': False}

        assert updater_mock._upsert_data.call_args_list[1].args == (updater_mock.feature_layer, 'old_data')
        assert updater_mock._upsert_data.call_args_list[1].kwargs == {'upsert': False}

        save_mock.assert_called_once_with('old_data', '/foo')

        assert 'Append failed; attempting to re-load truncated data...' in caplog.text
        assert 'features reloaded' not in caplog.text

    def test_truncate_and_load_raises_on_empty_column(self, mocker, caplog):

        caplog.set_level(logging.DEBUG)

        new_dataframe = pd.DataFrame({
            'id': [0, 1, 2],
            'floats': [np.nan, np.nan, np.nan],
            'x': [21, 22, 23],
            'y': [31, 32, 33]
        })
        spatial_df = pd.DataFrame.spatial.from_xy(new_dataframe, 'x', 'y')

        updater_mock = mocker.Mock()
        updater_mock.new_dataframe = spatial_df
        updater_mock._class_logger = logging.getLogger('mock logger')
        updater_mock.feature_service_itemid = 'foo123'
        updater_mock.layer_index = 0
        updater_mock.fields = ['id', 'floats', 'x', 'y']
        updater_mock._truncate_existing_data.return_value = 'old_data'

        updater_mock.feature_layer.properties = {
            'fields': [
                {
                    'name': 'id',
                    'type': 'esriFieldTypeInteger',
                    'nullable': True,
                    'defaultValue': None,
                },
                {
                    'name': 'floats',
                    'type': 'esriFieldTypeDouble',
                    'nullable': True,
                    'defaultValue': None,
                },
                {
                    'name': 'x',
                    'type': 'esriFieldTypeInteger',
                    'nullable': True,
                    'defaultValue': None,
                },
                {
                    'name': 'y',
                    'type': 'esriFieldTypeInteger',
                    'nullable': True,
                    'defaultValue': None,
                },
            ],
            'geometryType': 'esriGeometryPoint'
        }

        # mocker.patch('palletjack.utils.replace_nan_series_with_bogus_value', return_value='new_data')
        mocker.patch('palletjack.utils.sleep')
        # field_checker_mock = mocker.patch('palletjack.utils.FieldChecker')

        updater_mock._upsert_data.return_value = {'recordCount': 42}

        with pytest.raises(
            ValueError,
            match=re.escape(
                'The following float/double column(s) are completely empty: [\'floats\'] (suggestion: insert at least one bogus value)'
            )
        ):
            uploaded_features = load.FeatureServiceUpdater._truncate_and_load_data(updater_mock)

        # assert uploaded_features == 42


class TestAttachments:

    def test_create_attachment_action_df_adds_for_blank_existing_name(self, mocker):
        input_df = pd.DataFrame({
            'NAME': [np.nan],
            'new_path': ['bee/foo.png'],
        })

        ops_df = load.FeatureServiceAttachmentsUpdater._create_attachment_action_df(mocker.Mock(), input_df, 'new_path')

        test_df = pd.DataFrame({
            'NAME': [np.nan],
            'new_path': ['bee/foo.png'],
            'new_filename': ['foo.png'],
            'operation': ['add'],
        })

        tm.assert_frame_equal(ops_df, test_df)

    def test_create_attachment_action_df_overwrites_for_different_existing_name(self, mocker):
        input_df = pd.DataFrame({
            'NAME': ['bar.png'],
            'new_path': ['bee/foo.png'],
        })

        ops_df = load.FeatureServiceAttachmentsUpdater._create_attachment_action_df(mocker.Mock(), input_df, 'new_path')

        test_df = pd.DataFrame({
            'NAME': ['bar.png'],
            'new_path': ['bee/foo.png'],
            'new_filename': ['foo.png'],
            'operation': ['overwrite'],
        })

        tm.assert_frame_equal(ops_df, test_df)

    def test_create_attachment_action_df_does_nothing_for_same_name(self, mocker):
        input_df = pd.DataFrame({
            'NAME': ['foo.png'],
            'new_path': ['bee/foo.png'],
        })

        ops_df = load.FeatureServiceAttachmentsUpdater._create_attachment_action_df(mocker.Mock(), input_df, 'new_path')

        test_df = pd.DataFrame({
            'NAME': ['foo.png'],
            'new_path': ['bee/foo.png'],
            'new_filename': ['foo.png'],
            'operation': [np.nan],
        })
        test_df['operation'] = test_df['operation'].astype(object)

        tm.assert_frame_equal(ops_df, test_df)

    def test_create_attachment_action_df_does_all_three_ops(self, mocker):
        input_df = pd.DataFrame({
            'NAME': ['foo.png', 'bar.png', np.nan],
            'new_path': ['bee/foo.png', 'bee/baz.png', 'bee/bin.png'],
        })

        ops_df = load.FeatureServiceAttachmentsUpdater._create_attachment_action_df(mocker.Mock(), input_df, 'new_path')

        test_df = pd.DataFrame({
            'NAME': ['foo.png', 'bar.png', np.nan],
            'new_path': ['bee/foo.png', 'bee/baz.png', 'bee/bin.png'],
            'new_filename': ['foo.png', 'baz.png', 'bin.png'],
            'operation': [np.nan, 'overwrite', 'add'],
        })

        tm.assert_frame_equal(ops_df, test_df)

    def test_create_attachment_action_df_do_nothing_after_others(self, mocker):
        input_df = pd.DataFrame({
            'NAME': ['bar.png', np.nan, 'foo.png'],
            'new_path': ['bee/baz.png', 'bee/bin.png', 'bee/foo.png'],
        })

        ops_df = load.FeatureServiceAttachmentsUpdater._create_attachment_action_df(mocker.Mock(), input_df, 'new_path')

        test_df = pd.DataFrame({
            'NAME': ['bar.png', np.nan, 'foo.png'],
            'new_path': ['bee/baz.png', 'bee/bin.png', 'bee/foo.png'],
            'new_filename': ['baz.png', 'bin.png', 'foo.png'],
            'operation': ['overwrite', 'add', np.nan],
        })

        tm.assert_frame_equal(ops_df, test_df)

    def test_get_live_data_from_join_field_values_only_gets_matching_data(self, mocker):
        live_features_df = pd.DataFrame({
            'OBJECTID': [1, 2, 3],
            'GlobalID': ['guid1', 'guid2', 'guid3'],
            'attachment_key': [11, 12, 13],
            'deleted': ['a', 'b', 'c'],
        })

        attachments_df = pd.DataFrame({
            'attachment_key': [12, 13],
            'attachments': ['foo', 'bar'],
        })

        live_data_subset = load.FeatureServiceAttachmentsUpdater._get_live_oid_and_guid_from_join_field_values(
            mocker.Mock(), live_features_df, 'attachment_key', attachments_df
        )

        test_df = pd.DataFrame({
            'OBJECTID': [2, 3],
            'GlobalID': ['guid2', 'guid3'],
            'attachment_key': [12, 13],
            'attachments': ['foo', 'bar'],
        })

        tm.assert_frame_equal(live_data_subset, test_df)

    def test_get_current_attachment_info_by_oid_includes_nans_for_features_wo_attachments(self, mocker):

        live_attachments = [
            {
                'PARENTOBJECTID': 1,
                'PARENTGLOBALID': 'parentguid1',
                'ID': 111,
                'NAME': 'foo.png',
                'CONTENTTYPE': 'image/png',
                'SIZE': 42,
                'KEYWORDS': '',
                'IMAGE_PREVIEW': 'preview1',
                'GLOBALID': 'guid1',
                'DOWNLOAD_URL': 'url1'
            },
            {
                'PARENTOBJECTID': 2,
                'PARENTGLOBALID': 'parentguid2',
                'ID': 222,
                'NAME': 'bar.png',
                'CONTENTTYPE': 'image/png',
                'SIZE': 42,
                'KEYWORDS': '',
                'IMAGE_PREVIEW': 'preview2',
                'GLOBALID': 'guid2',
                'DOWNLOAD_URL': 'url2'
            },
        ]

        updater_mock = mocker.Mock()
        updater_mock.feature_layer.attachments.search.return_value = live_attachments

        live_data_subset_df = pd.DataFrame({
            'OBJECTID': [1, 2, 3],
            'GlobalID': ['guid1', 'guid2', 'guid3'],
            'attachment_key': [11, 12, 13],
            'attachments': ['fee', 'ber', 'boo'],
        })

        current_attachments_df = load.FeatureServiceAttachmentsUpdater._get_current_attachment_info_by_oid(
            updater_mock, live_data_subset_df
        )

        test_df = pd.DataFrame({
            'OBJECTID': [1, 2, 3],
            'GlobalID': ['guid1', 'guid2', 'guid3'],
            'attachment_key': [11, 12, 13],
            'attachments': ['fee', 'ber', 'boo'],
            'PARENTOBJECTID': [1., 2., np.nan],
            'NAME': ['foo.png', 'bar.png', np.nan],
            'ID': [111, 222, pd.NA],
        })
        test_df['ID'] = test_df['ID'].astype('Int64')

        tm.assert_frame_equal(current_attachments_df, test_df)

    def test_check_attachment_dataframe_for_invalid_column_names_doesnt_raise_with_valid_names(self, mocker):
        dataframe = pd.DataFrame(columns=['foo', 'bar'])
        invalid_names = ['baz', 'boo']
        load.FeatureServiceAttachmentsUpdater._check_attachment_dataframe_for_invalid_column_names(
            dataframe, invalid_names
        )

    def test_check_attachment_dataframe_for_invalid_column_names_raises_with_one_invalid(self, mocker):
        dataframe = pd.DataFrame(columns=['foo', 'bar'])
        invalid_names = ['foo', 'boo']
        with pytest.raises(RuntimeError) as exc_info:
            load.FeatureServiceAttachmentsUpdater._check_attachment_dataframe_for_invalid_column_names(
                dataframe, invalid_names
            )
        assert exc_info.value.args[0] == 'Attachment dataframe contains the following invalid names: [\'foo\']'

    def test_check_attachment_dataframe_for_invalid_column_names_raises_with_all_invalid(self, mocker):
        dataframe = pd.DataFrame(columns=['foo', 'bar'])
        invalid_names = ['foo', 'bar']
        with pytest.raises(RuntimeError) as exc_info:
            load.FeatureServiceAttachmentsUpdater._check_attachment_dataframe_for_invalid_column_names(
                dataframe, invalid_names
            )
        assert exc_info.value.args[0] == 'Attachment dataframe contains the following invalid names: [\'foo\', \'bar\']'

    def test_add_attachments_by_oid_adds_and_doesnt_warn(self, mocker):
        action_df = pd.DataFrame({
            'OBJECTID': [1, 2],
            'operation': ['add', 'add'],
            'path': ['path1', 'path2'],
        })

        result_dict = [
            {
                'addAttachmentResult': {
                    'success': True
                }
            },
            {
                'addAttachmentResult': {
                    'success': True
                }
            },
        ]

        updater_mock = mocker.Mock()
        updater_mock.feature_layer.attachments.add.side_effect = result_dict

        with pytest.warns(None) as warning:
            count = load.FeatureServiceAttachmentsUpdater._add_attachments_by_oid(updater_mock, action_df, 'path')

        assert count == 2
        assert not warning

    def test_add_attachments_by_oid_warns_on_failure_and_doesnt_count_that_one_and_continues(self, mocker):
        action_df = pd.DataFrame({
            'OBJECTID': [1, 2, 3],
            'operation': ['add', 'add', 'add'],
            'path': ['path1', 'path2', 'path3'],
        })

        result_dict = [
            {
                'addAttachmentResult': {
                    'success': True
                }
            },
            {
                'addAttachmentResult': {
                    'success': False
                }
            },
            {
                'addAttachmentResult': {
                    'success': True
                }
            },
        ]

        feature_layer_mock = mocker.Mock()
        feature_layer_mock.attachments.add.side_effect = result_dict

        updater = load.FeatureServiceAttachmentsUpdater(mocker.Mock())
        updater.feature_layer = feature_layer_mock

        with pytest.warns(UserWarning, match='Failed to attach path2 to OID 2'):
            count = updater._add_attachments_by_oid(action_df, 'path')

        assert count == 2
        assert updater.failed_dict == {2: ('add', 'path2')}

    def test_add_attachments_by_oid_handles_internal_agol_errors(self, mocker, caplog):
        action_df = pd.DataFrame({
            'OBJECTID': [1, 2],
            'operation': ['add', 'add'],
            'path': ['path1', 'path2'],
        })

        feature_layer_mock = mocker.Mock()
        feature_layer_mock.attachments.add.side_effect = [
            RuntimeError('foo'),
            {
                'addAttachmentResult': {
                    'success': True
                }
            },
        ]

        updater = load.FeatureServiceAttachmentsUpdater(mocker.Mock())
        updater.feature_layer = feature_layer_mock

        count = updater._add_attachments_by_oid(action_df, 'path')
        assert count == 1
        assert 'AGOL error while adding path1 to OID 1' in caplog.text
        assert 'foo' in caplog.text
        assert updater.failed_dict == {1: ('add', 'path1')}

    def test_add_attachments_by_oid_skips_overwrite_and_nan(self, mocker):
        action_df = pd.DataFrame({
            'OBJECTID': [1, 2, 3],
            'operation': ['add', 'overwrite', np.nan],
            'path': ['path1', 'path2', 'path3'],
        })

        result_dict = [
            {
                'addAttachmentResult': {
                    'success': True
                }
            },
        ]

        updater_mock = mocker.Mock()
        updater_mock.feature_layer.attachments.add.side_effect = result_dict

        count = load.FeatureServiceAttachmentsUpdater._add_attachments_by_oid(updater_mock, action_df, 'path')

        assert updater_mock.feature_layer.attachments.add.call_count == 1
        assert count == 1

    def test_overwrite_attachments_by_oid_overwrites_and_doesnt_warn(self, mocker):
        action_df = pd.DataFrame({
            'OBJECTID': [1, 2],
            'operation': ['overwrite', 'overwrite'],
            'path': ['path1', 'path2'],
            'ID': ['existing1', 'existing2'],
            'NAME': ['oldname1', 'oldname2'],
        })

        result_dict = [
            {
                'updateAttachmentResult': {
                    'success': True
                }
            },
            {
                'updateAttachmentResult': {
                    'success': True
                }
            },
        ]

        updater_mock = mocker.Mock()
        updater_mock.feature_layer.attachments.update.side_effect = result_dict

        with pytest.warns(None) as warning:
            count = load.FeatureServiceAttachmentsUpdater._overwrite_attachments_by_oid(updater_mock, action_df, 'path')

        assert count == 2
        assert not warning

    def test_overwrite_attachments_by_oid_warns_on_failure_and_doesnt_count_that_one_and_continues(self, mocker):
        action_df = pd.DataFrame({
            'OBJECTID': [1, 2, 3],
            'operation': ['overwrite', 'overwrite', 'overwrite'],
            'path': ['path1', 'path2', 'path3'],
            'ID': [11, 22, 33],
            'NAME': ['oldname1', 'oldname2', 'oldname3'],
        })

        result_dict = [
            {
                'updateAttachmentResult': {
                    'success': True
                }
            },
            {
                'updateAttachmentResult': {
                    'success': False
                }
            },
            {
                'updateAttachmentResult': {
                    'success': True
                }
            },
        ]

        feature_layer_mock = mocker.Mock()
        feature_layer_mock.attachments.update.side_effect = result_dict

        updater = load.FeatureServiceAttachmentsUpdater(mocker.Mock())
        updater.feature_layer = feature_layer_mock

        with pytest.warns(UserWarning, match='Failed to update oldname2, attachment ID 22, on OID 2 with path2'):
            count = updater._overwrite_attachments_by_oid(action_df, 'path')

        assert count == 2
        assert updater.failed_dict == {2: ('update', 'path2')}

    def test_overwrite_attachments_by_oid_handles_internal_agol_errors(self, mocker, caplog):
        action_df = pd.DataFrame({
            'OBJECTID': [1, 2],
            'operation': ['overwrite', 'overwrite'],
            'path': ['path1', 'path2'],
            'ID': [11, 22],
            'NAME': ['oldname1', 'oldname2'],
        })

        feature_layer_mock = mocker.Mock()
        feature_layer_mock.attachments.update.side_effect = [
            RuntimeError('foo'),
            {
                'updateAttachmentResult': {
                    'success': True
                }
            },
        ]

        updater = load.FeatureServiceAttachmentsUpdater(mocker.Mock())
        updater.feature_layer = feature_layer_mock

        count = updater._overwrite_attachments_by_oid(action_df, 'path')
        assert count == 1
        assert 'AGOL error while overwriting oldname1 (attachment ID 11) on OID 1 with path1' in caplog.text
        assert 'foo' in caplog.text
        assert updater.failed_dict == {1: ('update', 'path1')}

    def test_overwrite_attachments_by_oid_skips_add_and_nan(self, mocker):
        action_df = pd.DataFrame({
            'OBJECTID': [1, 2, 3],
            'operation': ['add', 'overwrite', np.nan],
            'path': ['path1', 'path2', 'path3'],
            'ID': [np.nan, 'existing2', 'existing3'],
            'NAME': ['oldname1', 'oldname2', 'oldname3'],
        })

        result_dict = [
            {
                'updateAttachmentResult': {
                    'success': True
                }
            },
        ]

        updater_mock = mocker.Mock()
        updater_mock.feature_layer.attachments.update.side_effect = result_dict

        count = load.FeatureServiceAttachmentsUpdater._overwrite_attachments_by_oid(updater_mock, action_df, 'path')

        assert updater_mock.feature_layer.attachments.update.call_count == 1
        assert count == 1

    def test_create_attachments_dataframe_subsets_and_crafts_paths_properly(self, mocker):
        input_df = pd.DataFrame({
            'join': [1, 2, 3],
            'pic': ['foo.png', 'bar.png', 'baz.png'],
            'data': [11., 12., 13.],
        })

        attachment_df = load.FeatureServiceAttachmentsUpdater.build_attachments_dataframe(
            input_df, 'join', 'pic', '/foo/bar'
        )

        test_df = pd.DataFrame({
            'join': [1, 2, 3],
            'pic': ['foo.png', 'bar.png', 'baz.png'],
            'full_file_path': [Path('/foo/bar/foo.png'),
                               Path('/foo/bar/bar.png'),
                               Path('/foo/bar/baz.png')]
        })

        #: Column of path objects won't test equal in assert_frame_equal, so we make lists of their str representations
        #: and compare those separately from the rest of the dataframe
        other_fields = ['join', 'pic']
        tm.assert_frame_equal(attachment_df[other_fields], test_df[other_fields])
        assert [str(path) for path in attachment_df['full_file_path']
               ] == [str(path) for path in test_df['full_file_path']]

    def test_create_attachments_dataframe_drops_missing_attachments(self, mocker):
        input_df = pd.DataFrame({
            'join': [1, 2, 3],
            'pic': ['foo.png', None, ''],
        })

        attachment_df = load.FeatureServiceAttachmentsUpdater.build_attachments_dataframe(
            input_df, 'join', 'pic', '/foo/bar'
        )

        test_df = pd.DataFrame({
            'join': [1],
            'pic': ['foo.png'],
            'full_file_path': [Path('/foo/bar/foo.png')],
        })

        #: Column of path objects won't test equal in assert_frame_equal, so we make lists of their str representations
        #: and compare those separately from the rest of the dataframe
        other_fields = ['join', 'pic']
        tm.assert_frame_equal(attachment_df[other_fields], test_df[other_fields])
        assert [str(path) for path in attachment_df['full_file_path']
               ] == [str(path) for path in test_df['full_file_path']]


class TestUpsertData:

    def test_upsert_data_calls_append_with_proper_args(self, mocker):
        mock_fl = mocker.Mock()
        mock_fl.append.return_value = (True, {'message': 'foo'})
        mock_df = mocker.Mock()
        mock_df.spatial.to_featureset.return_value.to_geojson = 'json'
        mocker.patch('palletjack.utils.sleep')

        load.FeatureServiceUpdater._upsert_data(mocker.Mock(), mock_fl, mock_df, upsert=True)

        mock_fl.append.assert_called_once_with(
            upload_format='geojson', edits='json', upsert=True, rollback=True, return_messages=True
        )

    def test_upsert_data_retries_on_exception(self, mocker):
        mock_fl = mocker.Mock()
        mock_fl.append.side_effect = [Exception, (True, {'message': 'foo'})]
        mock_df = mocker.Mock()
        mock_df.spatial.to_featureset.return_value.to_geojson = 'json'
        mocker.patch('palletjack.utils.sleep')

        load.FeatureServiceUpdater._upsert_data(mocker.Mock(), mock_fl, mock_df, upsert=True)

        assert mock_fl.append.call_count == 2

    def test_upsert_data_raises_on_False_result(self, mocker):
        mock_fl = mocker.Mock()
        mock_fl.append.return_value = (False, {'message': 'foo'})
        mock_df = mocker.Mock()
        mock_df.spatial.to_featureset.return_value.to_geojson = 'json'
        mocker.patch('palletjack.utils.sleep')

        with pytest.raises(RuntimeError) as exc_info:
            load.FeatureServiceUpdater._upsert_data(mocker.Mock(), mock_fl, mock_df, upsert=True)

        assert exc_info.value.args[0] == 'Failed to append data. Append operation should have been rolled back.'

    def test_upsert_data_raises_on_upsert_field_not_in_append_fields(self, mocker):
        append_kwargs = {'upsert_matching_field': 'foo', 'append_fields': ['bar', 'baz'], 'upsert': True}
        with pytest.raises(ValueError) as exc_info:
            load.FeatureServiceUpdater._upsert_data(mocker.Mock(), mocker.Mock(), mocker.Mock(), **append_kwargs)

        assert exc_info.value.args[0
                                  ] == 'Upsert matching field foo not found in either append fields or existing fields.'

    def test_upsert_data_raises_on_upsert_field_not_in_dataframe_columns(self, mocker):
        append_kwargs = {'upsert_matching_field': 'foo', 'append_fields': ['foo', 'bar'], 'upsert': True}
        df = pd.DataFrame(columns=['bar', 'baz'])
        with pytest.raises(ValueError) as exc_info:
            load.FeatureServiceUpdater._upsert_data(mocker.Mock(), mocker.Mock(), df, **append_kwargs)

        assert exc_info.value.args[0
                                  ] == 'Upsert matching field foo not found in either append fields or existing fields.'

    def test_upsert_data_raises_on_upsert_field_not_in_dataframe_columns_and_append_fields(self, mocker):
        append_kwargs = {'upsert_matching_field': 'foo', 'append_fields': ['bar', 'baz'], 'upsert': True}
        df = pd.DataFrame(columns=['bar', 'baz'])
        with pytest.raises(ValueError) as exc_info:
            load.FeatureServiceUpdater._upsert_data(mocker.Mock(), mocker.Mock(), df, **append_kwargs)

        assert exc_info.value.args[0
                                  ] == 'Upsert matching field foo not found in either append fields or existing fields.'


class TestColorRampReclassifier:

    def test_get_layer_id_returns_match_single_layer(self, mocker):
        layers = {
            'operationalLayers': [
                {
                    'title': 'foo'
                },
            ],
        }
        get_data_mock = mocker.Mock(return_value=layers)
        webmap_item_mock = mocker.Mock()
        webmap_item_mock.get_data = get_data_mock
        reclassifier = load.ColorRampReclassifier(webmap_item_mock, 'gis')

        layer_id = reclassifier._get_layer_id('foo')
        assert layer_id == 0

    def test_get_layer_id_returns_match_many_layers(self, mocker):
        layers = {
            'operationalLayers': [
                {
                    'title': 'foo'
                },
                {
                    'title': 'bar'
                },
                {
                    'title': 'baz'
                },
            ],
        }
        get_data_mock = mocker.Mock(return_value=layers)
        webmap_item_mock = mocker.Mock()
        webmap_item_mock.get_data = get_data_mock
        reclassifier = load.ColorRampReclassifier(webmap_item_mock, 'gis')

        layer_id = reclassifier._get_layer_id('bar')
        assert layer_id == 1

    def test_get_layer_id_returns_first_match(self, mocker):
        layers = {
            'operationalLayers': [
                {
                    'title': 'foo'
                },
                {
                    'title': 'bar'
                },
                {
                    'title': 'bar'
                },
            ],
        }
        get_data_mock = mocker.Mock(return_value=layers)
        webmap_item_mock = mocker.Mock()
        webmap_item_mock.get_data = get_data_mock
        reclassifier = load.ColorRampReclassifier(webmap_item_mock, 'gis')

        layer_id = reclassifier._get_layer_id('bar')
        assert layer_id == 1

    def test_get_layer_id_raises_error_when_not_found(self, mocker):
        layers = {
            'operationalLayers': [
                {
                    'title': 'bar'
                },
            ],
        }
        get_data_mock = mocker.Mock(return_value=layers)
        webmap_item_mock = mocker.Mock()
        webmap_item_mock.title = 'test map'
        webmap_item_mock.get_data = get_data_mock
        reclassifier = load.ColorRampReclassifier(webmap_item_mock, 'gis')

        with pytest.raises(ValueError) as error_info:
            layer_id = reclassifier._get_layer_id('foo')

        assert 'Could not find "foo" in test map' in str(error_info.value)

    def test_calculate_new_stops_with_manual_numbers(self):
        dataframe = pd.DataFrame({'numbers': [100, 300, 500, 700, 900]})

        stops = load.ColorRampReclassifier._calculate_new_stops(dataframe, 'numbers', 5)

        assert stops == [100, 279, 458, 637, 816]

    def test_calculate_new_stops_mismatched_column_raises_error(self):
        dataframe = pd.DataFrame({'numbers': [100, 300, 500, 700, 900]})

        with pytest.raises(ValueError) as error_info:
            stops = load.ColorRampReclassifier._calculate_new_stops(dataframe, 'foo', 5)
            assert 'Column `foo` not in dataframe`' in str(error_info)

    def test_update_stops_values(self, mocker):
        # renderer = data['operationalLayers'][layer_number]['layerDefinition']['drawingInfo']['renderer']
        # stops = renderer['visualVariables'][0]['stops']

        data = {
            'operationalLayers': [{
                'layerDefinition': {
                    'drawingInfo': {
                        'renderer': {
                            'visualVariables': [{
                                'stops': [{
                                    'value': 0
                                }, {
                                    'value': 1
                                }, {
                                    'value': 2
                                }, {
                                    'value': 3
                                }]
                            }]
                        }
                    }
                }
            }],
        }
        get_data_mock = mocker.Mock(return_value=data)
        webmap_item_mock = mocker.Mock()
        webmap_item_mock.get_data = get_data_mock
        update_mock = mocker.Mock()
        webmap_item_mock.update = update_mock
        reclassifier = load.ColorRampReclassifier(webmap_item_mock, 'gis')

        reclassifier._update_stop_values(0, [100, 200, 300, 400])

        data['operationalLayers'][0]['layerDefinition']['drawingInfo']['renderer']['visualVariables'][0]['stops'] = [{
            'value': 100
        }, {
            'value': 200
        }, {
            'value': 300
        }, {
            'value': 400
        }]

        assert update_mock.called_with(item_properties={'text': json.dumps(data)})
