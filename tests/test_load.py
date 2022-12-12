import json
import logging
import sys
import urllib
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest

from palletjack import load


class TestFeatureServiceUpdaterInit:

    def test_init_raises_on_missing_index_field(self, mocker):

        new_dataframe = pd.DataFrame(columns=['Foo_field', 'Bar'])
        # mocker.patch.object(pd.DataFrame, 'spatial')

        with pytest.raises(KeyError) as exc_info:
            updater = load.FeatureServiceUpdater(mocker.Mock(), new_dataframe, 'Baz')

        assert exc_info.value.args[0] == 'Index column Baz not found in dataframe columns'

    def test_init_renames_dataframe_columns(self, mocker):

        new_dataframe = pd.DataFrame(columns=['Foo field', 'Bar'])
        # mocker.patch.object(pd.DataFrame, 'spatial')

        updater = load.FeatureServiceUpdater(mocker.Mock(), new_dataframe, 'Bar')

        assert list(updater.new_dataframe.columns) == ['Foo_field', 'Bar']

    def test_init_renames_index_column(self, mocker):

        new_dataframe = pd.DataFrame(columns=['Foo field', 'Bar'])
        # mocker.patch.object(pd.DataFrame, 'spatial')

        updater = load.FeatureServiceUpdater(mocker.Mock(), new_dataframe, 'Foo field')

        assert updater.index_column == 'Foo_field'


class TestUpdateLayer:

    def test_update_hosted_feature_layer_calls_upsert(self, mocker):
        new_dataframe = pd.DataFrame({
            'foo': [1, 2],
            'bar': [3, 4],
            'OBJECTID': [11, 12],
        })
        updater_mock = mocker.Mock()
        updater_mock.feature_service_itemid = 'foo123'
        updater_mock.feature_layer = mocker.Mock()
        updater_mock.new_datframe = new_dataframe
        updater_mock.fields = ['foo', 'bar']
        updater_mock.join_column = 'OBJECTID'
        updater_mock.layer_index = 0

        updater_mock._upsert_data.return_value = {'recordCount': 1}

        field_checker_mock = mocker.patch('palletjack.utils.FieldChecker')

        load.FeatureServiceUpdater.update_hosted_feature_layer(updater_mock)

        assert updater_mock._upsert_data.called_once_with(
            'foo123', new_dataframe, upsert=True, upsert_matching_field='OBJECTID', append_fields=['foo', 'bar']
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
        updater_mock.new_datframe = new_dataframe
        updater_mock.fields = ['foo', 'bar']
        updater_mock.join_column = 'OBJECTID'
        updater_mock.layer_index = 0

        updater_mock._upsert_data.return_value = {'recordCount': 1}

        field_checker_mock = mocker.patch('palletjack.utils.FieldChecker')

        load.FeatureServiceUpdater.update_hosted_feature_layer(updater_mock)

        assert field_checker_mock.check_live_and_new_field_types_match.called_once_with(['foo', 'bar'])
        assert field_checker_mock.check_for_non_null_fields.called_once_with(['foo', 'bar'])
        assert field_checker_mock.check_field_length.called_once_with(['foo', 'bar'])
        assert field_checker_mock.check_fields_present.called_once_with(['foo', 'bar'], True)


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


class TestFeatureServiceOverwriter:

    def test_truncate_existing_data_normal(self, mocker):
        fl_mock = mocker.Mock()
        fl_mock.manager.truncate.return_value = {
            'submissionTime': 123,
            'lastUpdatedTime': 124,
            'status': 'Completed',
        }
        overwriter = load.FeatureServiceOverwriter(mocker.Mock())

        load.FeatureServiceOverwriter._truncate_existing_data(overwriter, fl_mock, 0, 'abc')

    def test_truncate_existing_raises_error_on_failure(self, mocker):
        fl_mock = mocker.Mock()
        fl_mock.manager.truncate.return_value = {
            'submissionTime': 123,
            'lastUpdatedTime': 124,
            'status': 'Foo',
        }
        overwriter = load.FeatureServiceOverwriter(mocker.Mock())

        with pytest.raises(RuntimeError) as exc_info:
            load.FeatureServiceOverwriter._truncate_existing_data(overwriter, fl_mock, 0, 'abc')

        assert exc_info.value.args[0] == 'Failed to truncate existing data from layer id 0 in itemid abc'

    def test_truncate_existing_retries_on_HTTPError(self, mocker):
        fl_mock = mocker.Mock()
        fl_mock.manager.truncate.side_effect = [
            urllib.error.HTTPError('url', 'code', 'msg', 'hdrs', 'fp'), {
                'submissionTime': 123,
                'lastUpdatedTime': 124,
                'status': 'Completed',
            }
        ]
        overwriter = load.FeatureServiceOverwriter(mocker.Mock())

        load.FeatureServiceOverwriter._truncate_existing_data(overwriter, fl_mock, 0, 'abc')

    def test_append_new_data_doesnt_raise_on_normal(self, mocker):
        mock_df = mocker.Mock()
        mock_fl = mocker.Mock()
        mock_fl.append.return_value = (True, {'message': 'foo'})

        overwriter = load.FeatureServiceOverwriter(mocker.Mock())
        overwriter._append_new_data(mock_fl, mock_df, 0, 'abc')

    def test_append_new_data_retries_on_httperror(self, mocker):
        mock_df = mocker.Mock()
        mock_fl = mocker.Mock()
        mock_fl.append.side_effect = [urllib.error.HTTPError('a', 'b', 'c', 'd', 'e'), (True, {'message': 'foo'})]

        overwriter = load.FeatureServiceOverwriter(mocker.Mock())
        overwriter._append_new_data(mock_fl, mock_df, 0, 'abc')

    def test_append_new_data_raises_on_False_result(self, mocker):
        mock_df = mocker.Mock()
        mock_fl = mocker.Mock()
        mock_fl.append.return_value = (False, {'message': 'foo'})

        overwriter = load.FeatureServiceOverwriter(mocker.Mock())

        with pytest.raises(RuntimeError) as exc_info:
            overwriter._append_new_data(mock_fl, mock_df, 'abc', 0)

        assert exc_info.value.args[
            0] == 'Failed to append data to layer id 0 in itemid abc. Append should have been rolled back.'

    def test_truncate_and_load_feature_service_normal(self, mocker):
        mock_fl = mocker.Mock()
        mock_fl.manager.truncate.return_value = {
            'submissionTime': 123,
            'lastUpdatedTime': 124,
            'status': 'Completed',
        }
        mock_fl.properties = {
            'fields': [
                {
                    'name': 'Foo'
                },
                {
                    'name': 'Bar'
                },
            ]
        }
        mock_fl.append.return_value = (True, {'recordCount': 42})

        fl_class_mock = mocker.Mock()
        fl_class_mock.fromitem.return_value = mock_fl
        mocker.patch('arcgis.features.FeatureLayer', fl_class_mock)

        new_dataframe = pd.DataFrame(columns=['Foo', 'Bar'])
        mocker.patch.object(pd.DataFrame, 'spatial')

        overwriter = load.FeatureServiceOverwriter(mocker.Mock())

        uploaded_features = overwriter.truncate_and_load_feature_service('abc', new_dataframe, 'foo/dir')

        assert uploaded_features == 42

    def test_truncate_and_load_append_fails_reload_works(self, mocker, caplog):
        caplog.set_level(logging.DEBUG)
        mock_fl = mocker.Mock()
        mock_fl.manager.truncate.return_value = {
            'submissionTime': 123,
            'lastUpdatedTime': 124,
            'status': 'Completed',
        }
        mock_fl.properties = {
            'fields': [
                {
                    'name': 'Foo'
                },
                {
                    'name': 'Bar'
                },
            ]
        }
        mock_fl.append.side_effect = [(False, 'foo'), (True, {'recordCount': 42})]

        fl_class_mock = mocker.Mock()
        fl_class_mock.fromitem.return_value = mock_fl
        mocker.patch('arcgis.features.FeatureLayer', fl_class_mock)

        new_dataframe = pd.DataFrame(columns=['Foo', 'Bar'])
        mocker.patch.object(pd.DataFrame, 'spatial')

        overwriter = load.FeatureServiceOverwriter(mocker.Mock())

        with pytest.raises(
            RuntimeError,
            match='Failed to append data to layer id 0 in itemid abc. Append should have been rolled back.'
        ):
            uploaded_features = overwriter.truncate_and_load_feature_service('abc', new_dataframe, 'foo/dir')

        assert 'Append failed; attempting to re-load truncated data...' in caplog.text
        assert '42 features reloaded' in caplog.text

    def test_truncate_and_load_field_check_fails_reload_works(self, mocker, caplog):
        caplog.set_level(logging.DEBUG)
        mock_fl = mocker.Mock()
        mock_fl.manager.truncate.return_value = {
            'submissionTime': 123,
            'lastUpdatedTime': 124,
            'status': 'Completed',
        }
        mock_fl.properties = {
            'fields': [
                {
                    'name': 'Foo'
                },
                {
                    'name': 'Bar'
                },
            ]
        }
        mock_fl.append.return_value = (True, {'recordCount': 42})

        fl_class_mock = mocker.Mock()
        fl_class_mock.fromitem.return_value = mock_fl
        mocker.patch('arcgis.features.FeatureLayer', fl_class_mock)

        new_dataframe = pd.DataFrame(columns=['Foo', 'Bar', 'Baz'])
        mocker.patch.object(pd.DataFrame, 'spatial')

        overwriter = load.FeatureServiceOverwriter(mocker.Mock())

        with pytest.raises(
            RuntimeError,
            match='New dataset contains the following fields that are not present in the live dataset: {\'Baz\'}'
        ):
            uploaded_features = overwriter.truncate_and_load_feature_service('abc', new_dataframe, 'foo/dir')

        assert 'Append failed; attempting to re-load truncated data...' in caplog.text
        assert '42 features reloaded' in caplog.text

    def test_truncate_and_load_append_fails_reload_fails(self, mocker, caplog):
        caplog.set_level(logging.DEBUG)
        mock_fl = mocker.Mock()
        mock_fl.manager.truncate.return_value = {
            'submissionTime': 123,
            'lastUpdatedTime': 124,
            'status': 'Completed',
        }
        mock_fl.properties = {
            'fields': [
                {
                    'name': 'Foo'
                },
                {
                    'name': 'Bar'
                },
            ]
        }
        mock_fl.append.side_effect = [(False, 'foo'), (False, 'bar')]

        fl_class_mock = mocker.Mock()
        fl_class_mock.fromitem.return_value = mock_fl
        mocker.patch('arcgis.features.FeatureLayer', fl_class_mock)

        new_dataframe = pd.DataFrame(columns=['Foo', 'Bar'])
        mocker.patch.object(pd.DataFrame, 'spatial')

        mocker.patch.object(load.FeatureServiceOverwriter, '_save_truncated_data', return_value='/foo/bar.json')

        overwriter = load.FeatureServiceOverwriter(mocker.Mock())

        with pytest.raises(RuntimeError) as exc_info:
            uploaded_features = overwriter.truncate_and_load_feature_service('abc', new_dataframe, 'foo/dir')

        assert exc_info.value.args[
            0] == 'Failed to re-add truncated data after failed append; data saved to /foo/bar.json'
        assert 'Append failed; attempting to re-load truncated data...' in caplog.text
        assert 'features reloaded' not in caplog.text

    def test_save_truncated_data_calls_write_with_json(self, mocker):
        mock_df = pd.DataFrame({
            'foo': [1],
            'x': [11],
            'y': [14],
        })
        #: Need to 'unload' the mocked arcpy used in tests above so that .from_xy() works. This could be pulled
        #: out if these tests or the other tests are in their own files.
        if 'arcpy' in sys.modules:
            del sys.modules['arcpy']

        mock_sdf = pd.DataFrame.spatial.from_xy(mock_df, 'x', 'y')

        open_mock = mocker.MagicMock()
        context_manager_mock = mocker.MagicMock()
        context_manager_mock.return_value.__enter__.return_value = open_mock
        mocker.patch('pathlib.Path.open', new=context_manager_mock)

        load.FeatureServiceOverwriter._save_truncated_data(mocker.Mock(), mock_sdf, 'foo')

        test_json_string = '{"features": [{"geometry": {"spatialReference": {"wkid": 4326}, "x": 11, "y": 14}, "attributes": {"foo": 1, "x": 11, "y": 14, "OBJECTID": 1}}], "objectIdFieldName": "OBJECTID", "displayFieldName": "OBJECTID", "spatialReference": {"wkid": 4326}, "geometryType": "esriGeometryPoint", "fields": [{"name": "OBJECTID", "type": "esriFieldTypeOID", "alias": "OBJECTID"}, {"name": "foo", "type": "esriFieldTypeInteger", "alias": "foo"}, {"name": "x", "type": "esriFieldTypeInteger", "alias": "x"}, {"name": "y", "type": "esriFieldTypeInteger", "alias": "y"}]}'

        assert open_mock.write.called_with(test_json_string)

    def test_save_truncated_data_opens_file_with_right_name(self, mocker):
        mock_df = pd.DataFrame({
            'foo': [1],
            'x': [11],
            'y': [14],
        })
        #: Need to 'unload' the mocked arcpy used in tests above so that .from_xy() works. This could be pulled
        #: out if these tests or the other tests are in their own files.
        if 'arcpy' in sys.modules:
            del sys.modules['arcpy']

        mock_sdf = pd.DataFrame.spatial.from_xy(mock_df, 'x', 'y')

        open_mock = mocker.MagicMock()
        context_manager_mock = mocker.MagicMock()
        context_manager_mock.return_value.__enter__.return_value = open_mock

        datetime_mock = mocker.Mock()
        datetime_mock.date.today.return_value = 'foo-date'
        mocker.patch('palletjack.load.datetime', new=datetime_mock)

        open_mock = mocker.MagicMock()
        context_manager_mock = mocker.MagicMock()
        context_manager_mock.return_value.__enter__.return_value = open_mock
        mocker.patch('pathlib.Path.open', new=context_manager_mock)

        out_path = load.FeatureServiceOverwriter._save_truncated_data(mocker.Mock(), mock_sdf, 'foo')

        assert out_path == Path('foo/old_data_foo-date.json')


class TestFeatureServiceUpdaterUpsert:

    def test_upsert_data_calls_append_with_proper_args(self, mocker):
        mock_fl = mocker.Mock()
        mock_fl.append.return_value = (True, {'message': 'foo'})
        mock_df = mocker.Mock()
        mock_df.spatial.to_featureset.return_value.to_geojson = 'json'
        mocker.patch('palletjack.utils.sleep')

        load.FeatureServiceUpdater._upsert_data(mocker.Mock(), mock_fl, mock_df, upsert=True)

        mock_fl.append.assert_called_with(
            upload_format='geojson', edits='json', upsert=True, rollback=True, return_messages=True
        )

    def test_upsert_data_doesnt_raise_on_normal(self, mocker):
        mock_df = mocker.Mock()
        mock_fl = mocker.Mock()
        mock_fl.append.return_value = (True, {'message': 'foo'})
        mocker.patch('palletjack.utils.rename_columns_for_agol')
        updater = load.FeatureServiceUpdater(mocker.Mock(), mocker.Mock(), 'foo')

        updater._upsert_data(mock_fl, mock_df)

    def test_upsert_data_retries_on_httperror(self, mocker):
        mock_df = mocker.Mock()
        mock_fl = mocker.Mock()
        mock_fl.append.side_effect = [Exception, (True, {'message': 'foo'})]
        mocker.patch('palletjack.utils.rename_columns_for_agol')

        updater = load.FeatureServiceUpdater(mocker.Mock(), mocker.Mock(), 'foo')
        updater._upsert_data(mock_fl, mock_df)

    def test_upsert_data_raises_on_False_result(self, mocker):
        mock_df = mocker.Mock()
        mock_fl = mocker.Mock()
        mock_fl.append.return_value = (False, {'message': 'foo'})
        mocker.patch('palletjack.utils.rename_columns_for_agol')

        updater = load.FeatureServiceUpdater(mocker.Mock(), mocker.Mock(), 'foo')

        with pytest.raises(RuntimeError) as exc_info:
            updater._upsert_data(mock_fl, mock_df)

        assert exc_info.value.args[0] == 'Failed to append data. Append operation should have been rolled back.'

    def test_append_new_data_to_hosted_feature_layer_normal(self, mocker):
        mock_fl = mocker.Mock()
        mock_fl.properties = {
            'fields': [
                {
                    'name': 'Foo'
                },
                {
                    'name': 'Bar'
                },
            ],
            'indexes': [
                {
                    'fields': 'Foo',
                    'isUnique': True
                },
            ]
        }

        mock_fl.append.return_value = (True, {'recordCount': 42})

        fl_class_mock = mocker.Mock()
        fl_class_mock.fromitem.return_value = mock_fl
        mocker.patch('arcgis.features.FeatureLayer', fl_class_mock)

        new_dataframe = pd.DataFrame(columns=['Foo', 'Bar'])
        mocker.patch.object(pd.DataFrame, 'spatial')

        updater = load.FeatureServiceUpdater(mocker.Mock(), new_dataframe, 'Foo')

        uploaded_features = updater.append_new_data_to_hosted_feature_layer('abc')

        assert uploaded_features == 42

    def test_append_new_data_to_hosted_feature_layer_handles_agol_field_renaming(self, mocker):
        mock_fl = mocker.Mock()
        mock_fl.properties = {
            'fields': [
                {
                    'name': 'Foo_field'
                },
                {
                    'name': 'Bar'
                },
            ],
            'indexes': [
                {
                    'fields': 'Bar',
                    'isUnique': True
                },
            ]
        }
        mock_fl.append.return_value = (True, {'recordCount': 42})

        fl_class_mock = mocker.Mock()
        fl_class_mock.fromitem.return_value = mock_fl
        mocker.patch('arcgis.features.FeatureLayer', fl_class_mock)

        new_dataframe = pd.DataFrame(columns=['Foo field', 'Bar'])
        mocker.patch.object(pd.DataFrame, 'spatial')

        updater = load.FeatureServiceUpdater(mocker.Mock(), new_dataframe, 'Bar')

        uploaded_features = updater.append_new_data_to_hosted_feature_layer('abc')

        assert uploaded_features == 42

    def test_append_new_data_to_hosted_feature_layer_handles_manual_field_renaming(self, mocker):
        mock_fl = mocker.Mock()

        mock_fl.properties = {
            'fields': [
                {
                    'name': 'Shape__Length'
                },
                {
                    'name': 'Bar'
                },
            ],
            'indexes': [
                {
                    'fields': 'Bar',
                    'isUnique': True
                },
            ]
        }
        mock_fl.append.return_value = (True, {'recordCount': 42})

        fl_class_mock = mocker.Mock()
        fl_class_mock.fromitem.return_value = mock_fl
        mocker.patch('arcgis.features.FeatureLayer', fl_class_mock)

        new_dataframe = pd.DataFrame(columns=['st_length(shape)', 'Bar'])
        field_mapping = {'st_length(shape)': 'Shape__Length'}
        mocker.patch.object(pd.DataFrame, 'spatial')

        updater = load.FeatureServiceUpdater(mocker.Mock(), new_dataframe, 'Bar', field_mapping=field_mapping)

        uploaded_features = updater.append_new_data_to_hosted_feature_layer('abc')

        assert uploaded_features == 42

    def test_upsert_existing_data_in_hosted_feature_layer_autogen_OIDs_dont_get_added(self, mocker):

        mock_fl = mocker.Mock()
        mock_fl.properties = {
            'fields': [
                {'name': 'OBJECTID'},
                {'name': 'data'},
                {'name': 'key'},
            ],
            'indexes': [{'fields': 'key', 'isUnique': True},]
        }  # yapf: disable
        mocker.patch.object(load.arcgis.features.FeatureLayer, 'fromitem', return_value=mock_fl)
        pd_mock = mocker.patch.object(load.pd.DataFrame.spatial, 'from_layer')
        pd_mock.return_value = pd.DataFrame({
            'OBJECTID': [14, 15],
            'data': ['foo', 'bar'],
            'key': ['a', 'b'],
        })

        upserter_mock = mocker.patch.object(load.FeatureServiceUpdater, '_upsert_data')

        new_dataframe = pd.DataFrame({
            'data': ['FOO', 'BAR'],
            'key': ['a', 'b'],
        })

        updater = load.FeatureServiceUpdater(mocker.Mock(), new_dataframe, 'key')

        updater.upsert_existing_features_in_hosted_feature_layer('abc', ['data', 'key'])

        test_df = pd.DataFrame({
            'key': ['a', 'b'],
            'data': ['FOO', 'BAR'],
        })

        assert upserter_mock.call_args.args[0] == mock_fl
        tm.assert_frame_equal(upserter_mock.call_args.args[1], test_df, check_like=True)
        assert upserter_mock.call_args.kwargs == {
            'upsert': True,
            'upsert_matching_field': 'key',
            'append_fields': ['data', 'key']
        }

    def test_upsert_existing_data_in_hosted_feature_layer_subsets_fields_in_live_df(self, mocker):

        mocker.patch.object(load.arcgis.features.FeatureLayer, 'fromitem')

        pd_mock = mocker.patch.object(load.pd.DataFrame.spatial, 'from_layer')
        pd_mock.return_value = pd.DataFrame({
            'OBJECTID': [1, 2],
            'data': ['foo', 'bar'],
            'key': ['a', 'b'],
        })

        new_dataframe = pd.DataFrame({
            'data': ['FOO', 'BAR'],
            'key': ['c', 'd'],
            'extra_column': ['baz', 'boo'],
        })
        mocker.patch.object(load.FeatureServiceUpdater, '_validate_working_fields_in_live_and_new_dataframes')
        mocker.patch.object(load.FeatureServiceUpdater, '_get_common_rows')
        mocker.patch.object(load.FeatureServiceUpdater, '_upsert_data')
        mocker.patch.object(load.utils, 'check_index_column_in_feature_layer')
        mocker.patch.object(load.utils, 'check_field_set_to_unique')

        updater = load.FeatureServiceUpdater(mocker.Mock(), new_dataframe, 'key')

        updater.upsert_existing_features_in_hosted_feature_layer('abc', ['data', 'key'])

        test_df = pd.DataFrame({
            'data': ['FOO', 'BAR'],
            'key': ['c', 'd'],
        })

        tm.assert_frame_equal(updater.new_dataframe, test_df)

    def test_upsert_existing_data_in_hosted_feature_layer_raises_on_missing_fields(self, mocker):

        mock_fl = mocker.Mock()
        mock_fl.properties = {
            'fields': [
                {'name': 'OBJECTID'},
                {'name': 'data'},
                {'name': 'key'},
            ],
            'indexes': [{'fields': 'key', 'isUnique': True},]
        }  # yapf: disable
        mocker.patch.object(load.arcgis.features.FeatureLayer, 'fromitem', return_value=mock_fl)
        pd_mock = mocker.patch.object(load.pd.DataFrame.spatial, 'from_layer')
        pd_mock.return_value = pd.DataFrame({
            'OBJECTID': [1, 2],
            'data': ['foo', 'bar'],
            'key': ['a', 'b'],
        })

        new_dataframe = pd.DataFrame({
            'data': ['FOO', 'BAR'],
            'key': ['a', 'b'],
        })

        updater = load.FeatureServiceUpdater(mocker.Mock(), new_dataframe, 'key')

        with pytest.raises(RuntimeError) as error_info:
            updater.upsert_existing_features_in_hosted_feature_layer('abc', ['data', 'KEY'])

        assert "Field mismatch between defined fields and either new or live data.\nFields not in live data: {'KEY'}\nFields not in new data: {'KEY'}" in str(
            error_info.value
        )

    def test_upsert_existing_data_in_hosted_feature_layer_only_includes_rows_from_new_data(self, mocker):
        mock_fl = mocker.Mock()
        mock_fl.properties = {
            'fields': [
                {'name': 'OBJECTID'},
                {'name': 'data'},
                {'name': 'key'},
            ],
            'indexes': [
                {'fields': 'key', 'isUnique': True},
            ]
        }  # yapf: disable
        mocker.patch.object(load.arcgis.features.FeatureLayer, 'fromitem', return_value=mock_fl)
        mocker.patch.object(
            load.pd.DataFrame.spatial,
            'from_layer',
            return_value=pd.DataFrame({
                'OBJECTID': [1, 2],
                'data': ['foo', 'bar'],
                'key': ['a', 'b'],
            })
        )

        upserter_mock = mocker.patch.object(load.FeatureServiceUpdater, '_upsert_data')

        new_dataframe = pd.DataFrame({
            'data': ['FOO', 'BAR'],
            'key': ['x', 'b'],
        })

        updater = load.FeatureServiceUpdater(mocker.Mock(), new_dataframe, 'key')

        updater.upsert_existing_features_in_hosted_feature_layer('abc', ['data', 'key'])

        test_df = pd.DataFrame({
            'key': ['b'],
            'data': ['BAR'],
        })

        assert upserter_mock.call_args.args[0] == mock_fl
        tm.assert_frame_equal(upserter_mock.call_args.args[1], test_df, check_like=True)
        assert upserter_mock.call_args.kwargs == {
            'upsert': True,
            'upsert_matching_field': 'key',
            'append_fields': ['data', 'key']
        }


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
