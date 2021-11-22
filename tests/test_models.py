import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from arcgis.features import GeoAccessor, GeoSeriesAccessor
from mock_arcpy import arcpy
from pandas.api.types import CategoricalDtype

import palletjack


class TestFeatureServiceInLineUpdater:

    def test_update_existing_features_in_feature_service_with_arcpy(self, mocker):
        #: We create a mock that will be returned by UpdateCursor's mock's __enter__, thus becoming our context manager.
        #: We then set it's __iter__.return_value to define the data we want it to iterate over.
        cursor_mock = mocker.MagicMock()
        cursor_mock.__iter__.return_value = [
            ['12345', '42', 123.45, '12/25/2021'],
            ['67890', '18', 67.89, '12/25/2021'],
        ]
        context_manager_mock = mocker.MagicMock()
        context_manager_mock.return_value.__enter__.return_value = cursor_mock
        mocker.patch('arcpy.da.UpdateCursor', new=context_manager_mock)

        fsupdater_mock = mocker.Mock()
        fsupdater_mock.data_as_dict = {'12345': {'Count': '57', 'Amount': 100.00, 'Date': '1/1/2022'}}

        palletjack.FeatureServiceInLineUpdater.update_existing_features_in_feature_service_with_arcpy(
            fsupdater_mock, 'foo', ['ZipCode', 'Count', 'Amount', 'Date']
        )

        cursor_mock.updateRow.assert_called_with(['12345', '57', 100.00, '1/1/2022'])
        cursor_mock.updateRow.assert_called_once()

    def test_clean_dataframe_columns_renames_and_drops_columns(self, mocker):
        class_mock = mocker.Mock()
        fields = ['foo', 'bar']
        dataframe = pd.DataFrame(columns=['foo_x', 'bar_x', 'baz', 'foo_y', 'bar_y'])

        renamed = palletjack.FeatureServiceInLineUpdater._clean_dataframe_columns(class_mock, dataframe, fields)

        assert list(renamed.columns) == ['baz', 'foo', 'bar']

    def test_clean_dataframe_columns_handles_field_not_found_in_dataframe(self, mocker):
        class_mock = mocker.Mock()
        fields = ['foo', 'bar', 'buz']
        dataframe = pd.DataFrame(columns=['foo_x', 'bar_x', 'baz', 'foo_y', 'bar_y'])

        renamed = palletjack.FeatureServiceInLineUpdater._clean_dataframe_columns(class_mock, dataframe, fields)

        assert list(renamed.columns) == ['baz', 'foo', 'bar']

    def test_clean_dataframe_columns_deletes_merge_field(self, mocker):
        class_mock = mocker.Mock()
        fields = ['foo', 'bar', 'buz']
        dataframe = pd.DataFrame(columns=['foo_x', 'bar_x', 'baz', 'foo_y', 'bar_y', '_merge'])

        renamed = palletjack.FeatureServiceInLineUpdater._clean_dataframe_columns(class_mock, dataframe, fields)

        assert list(renamed.columns) == ['baz', 'foo', 'bar']

    def test_get_common_rows_joins_properly_all_rows_in_both(self, mocker):
        class_mock = mocker.Mock()
        class_mock.index_column = 'key'
        class_mock.new_dataframe = pd.DataFrame({'col1': [10, 20, 30], 'col2': [40, 50, 60], 'key': ['a', 'b', 'c']})
        live_dataframe = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'key': ['a', 'b', 'c']})

        joined = palletjack.FeatureServiceInLineUpdater._get_common_rows(class_mock, live_dataframe)

        merge_type = CategoricalDtype(categories=['left_only', 'right_only', 'both'], ordered=False)
        expected = pd.DataFrame({
            'col1_x': [1, 2, 3],
            'col2_x': [4, 5, 6],
            'key': ['a', 'b', 'c'],
            'col1_y': [10, 20, 30],
            'col2_y': [40, 50, 60],
            '_merge': ['both', 'both', 'both']
        })
        expected['_merge'] = expected['_merge'].astype(merge_type)

        pd.testing.assert_frame_equal(joined, expected)

    def test_get_common_rows_subsets_properly_new_has_fewer_rows(self, mocker):

        class_mock = mocker.Mock()
        class_mock.index_column = 'key'
        class_mock.new_dataframe = pd.DataFrame({'col1': [20, 30], 'col2': [50, 60], 'key': ['b', 'c']})
        live_dataframe = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'key': ['a', 'b', 'c']})

        joined = palletjack.FeatureServiceInLineUpdater._get_common_rows(class_mock, live_dataframe)

        merge_type = CategoricalDtype(categories=['left_only', 'right_only', 'both'], ordered=False)
        expected = pd.DataFrame({
            'col1_x': [2, 3],
            'col2_x': [5, 6],
            'key': ['b', 'c'],
            'col1_y': [20, 30],
            'col2_y': [50, 60],
            '_merge': ['both', 'both']
        },
                                index=[1, 2])
        expected['_merge'] = expected['_merge'].astype(merge_type)

        pd.testing.assert_frame_equal(joined, expected, check_dtype=False)

    def test_get_common_rows_logs_warning_for_rows_not_in_existing_dataset(self, mocker, caplog):

        class_mock = mocker.Mock()
        class_mock.index_column = 'key'
        class_mock.new_dataframe = pd.DataFrame({
            'col1': [10, 20, 30, 80],
            'col2': [40, 50, 60, 70],
            'key': ['a', 'b', 'c', 'd']
        })
        class_mock._class_logger = logging.getLogger('root')
        live_dataframe = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'key': ['a', 'b', 'c']})

        joined = palletjack.FeatureServiceInLineUpdater._get_common_rows(class_mock, live_dataframe)

        merge_type = CategoricalDtype(categories=['left_only', 'right_only', 'both'], ordered=False)
        expected = pd.DataFrame({
            'col1_x': [1, 2, 3],
            'col2_x': [4, 5, 6],
            'key': ['a', 'b', 'c'],
            'col1_y': [10, 20, 30],
            'col2_y': [40, 50, 60],
            '_merge': ['both', 'both', 'both']
        },
                                index=[0, 1, 2])
        expected['_merge'] = expected['_merge'].astype(merge_type)

        pd.testing.assert_frame_equal(joined, expected, check_dtype=False)
        assert 'The following keys from the new data were not found in the existing dataset: [\'d\']' in caplog.text

    def test_parse_results_returns_correct_number_of_updated_rows(self, mocker):
        class_mock = mocker.Mock()
        results_dict = {
            'addResults': [],
            'updateResults': [
                {
                    'objectId': 1,
                    'success': True
                },
                {
                    'objectId': 2,
                    'success': True
                },
            ],
            'deleteResults': [],
        }
        live_dataframe = pd.DataFrame({
            'objectId': [1, 2],
            'data': ['foo', 'bar'],
        })

        rows_updated = palletjack.FeatureServiceInLineUpdater._parse_results(class_mock, results_dict, live_dataframe)

        assert rows_updated == 2

    def test_parse_results_retuns_0_if_both_successful_and_failed_result(self, mocker):
        class_mock = mocker.Mock()
        results_dict = {
            'addResults': [],
            'updateResults': [
                {
                    'objectId': 1,
                    'success': True
                },
                {
                    'objectId': 2,
                    'success': False
                },
            ],
            'deleteResults': [],
        }
        live_dataframe = pd.DataFrame({
            'objectId': [1, 2],
            'data': ['foo', 'bar'],
        })

        rows_updated = palletjack.FeatureServiceInLineUpdater._parse_results(class_mock, results_dict, live_dataframe)

        assert rows_updated == 0

    def test_parse_results_logs_success_info(self, mocker, caplog):
        class_mock = mocker.Mock()
        class_mock._class_logger = logging.getLogger('root')
        results_dict = {
            'addResults': [],
            'updateResults': [
                {
                    'objectId': 1,
                    'success': True
                },
                {
                    'objectId': 2,
                    'success': True
                },
            ],
            'deleteResults': [],
        }
        live_dataframe = pd.DataFrame({
            'objectId': [1, 2],
            'data': ['foo', 'bar'],
        })

        with caplog.at_level(logging.INFO):
            rows_updated = palletjack.FeatureServiceInLineUpdater._parse_results(
                class_mock, results_dict, live_dataframe
            )

            assert '2 rows successfully updated' in caplog.text
            assert "{'objectId': 1, 'data': 'foo'}" not in caplog.text
            assert "{'objectId': 2, 'data': 'bar'}" not in caplog.text

    def test_parse_results_logs_success_debug(self, mocker, caplog):
        class_mock = mocker.Mock()
        class_mock._class_logger = logging.getLogger('root')
        results_dict = {
            'addResults': [],
            'updateResults': [
                {
                    'objectId': 1,
                    'success': True
                },
                {
                    'objectId': 2,
                    'success': True
                },
            ],
            'deleteResults': [],
        }
        live_dataframe = pd.DataFrame({
            'objectId': [1, 2],
            'data': ['foo', 'bar'],
        })

        with caplog.at_level(logging.DEBUG):
            rows_updated = palletjack.FeatureServiceInLineUpdater._parse_results(
                class_mock, results_dict, live_dataframe
            )

            assert '2 rows successfully updated' in caplog.text
            assert "{'objectId': 1, 'data': 'foo'}" in caplog.text
            assert "{'objectId': 2, 'data': 'bar'}" in caplog.text

    def test_parse_results_logs_failures_at_warning(self, mocker, caplog):
        class_mock = mocker.Mock()
        class_mock._class_logger = logging.getLogger('root')
        results_dict = {
            'addResults': [],
            'updateResults': [
                {
                    'objectId': 1,
                    'success': True
                },
                {
                    'objectId': 2,
                    'success': False
                },
            ],
            'deleteResults': [],
        }
        live_dataframe = pd.DataFrame({
            'objectId': [1, 2],
            'data': ['foo', 'bar'],
        })

        with caplog.at_level(logging.WARNING):
            rows_updated = palletjack.FeatureServiceInLineUpdater._parse_results(
                class_mock, results_dict, live_dataframe
            )

            assert caplog.records[
                0
            ].message == 'The following 1 updates failed. As a result, all successfull updates should have been rolled back.' and caplog.records[
                0].levelname == 'WARNING'
            assert caplog.records[1].message == "{'objectId': 2, 'data': 'bar'}" and caplog.records[
                1].levelname == 'WARNING'

    def test_parse_results_doesnt_log_any_success_on_all_failed(self, mocker, caplog):
        class_mock = mocker.Mock()
        class_mock._class_logger = logging.getLogger('root')
        results_dict = {
            'addResults': [],
            'updateResults': [
                {
                    'objectId': 1,
                    'success': False
                },
                {
                    'objectId': 2,
                    'success': False
                },
            ],
            'deleteResults': [],
        }
        live_dataframe = pd.DataFrame({
            'objectId': [1, 2],
            'data': ['foo', 'bar'],
        })

        with caplog.at_level(logging.INFO):
            rows_updated = palletjack.FeatureServiceInLineUpdater._parse_results(
                class_mock, results_dict, live_dataframe
            )

            assert 'rows successfully updated' not in caplog.text

    def test_parse_results_logs_successes_before_failure(self, mocker, caplog):
        class_mock = mocker.Mock()
        class_mock._class_logger = logging.getLogger('root')
        results_dict = {
            'addResults': [],
            'updateResults': [
                {
                    'objectId': 1,
                    'success': True
                },
                {
                    'objectId': 2,
                    'success': False
                },
            ],
            'deleteResults': [],
        }
        live_dataframe = pd.DataFrame({
            'objectId': [1, 2],
            'data': ['foo', 'bar'],
        })

        with caplog.at_level(logging.DEBUG):
            rows_updated = palletjack.FeatureServiceInLineUpdater._parse_results(
                class_mock, results_dict, live_dataframe
            )
            assert caplog.records[0].message == '1 rows successfully updated' and caplog.records[0].levelname == 'INFO'
            assert caplog.records[2].message == "{'objectId': 1, 'data': 'foo'}" and caplog.records[
                2].levelname == 'DEBUG'
            assert caplog.records[
                3
            ].message == 'The following 1 updates failed. As a result, all successfull updates should have been rolled back.' and caplog.records[
                3].levelname == 'WARNING'
            assert caplog.records[4].message == "{'objectId': 2, 'data': 'bar'}" and caplog.records[
                4].levelname == 'WARNING'

    def test_parse_results_returns_0_when_results_are_empty(self, mocker, caplog):
        class_mock = mocker.Mock()
        results_dict = {
            'addResults': [],
            'updateResults': [],
            'deleteResults': [],
        }
        live_dataframe = pd.DataFrame({
            'objectId': [1, 2],
            'data': ['foo', 'bar'],
        })

        with caplog.at_level(logging.INFO):
            rows_updated = palletjack.FeatureServiceInLineUpdater._parse_results(
                class_mock, results_dict, live_dataframe
            )

            assert rows_updated == 0

    def test_parse_results_logs_appropriately_when_results_are_empty(self, mocker, caplog):
        class_mock = mocker.Mock()
        class_mock._class_logger = logging.getLogger('root')
        results_dict = {
            'addResults': [],
            'updateResults': [],
            'deleteResults': [],
        }
        live_dataframe = pd.DataFrame({
            'objectId': [1, 2],
            'data': ['foo', 'bar'],
        })

        with caplog.at_level(logging.INFO):
            rows_updated = palletjack.FeatureServiceInLineUpdater._parse_results(
                class_mock, results_dict, live_dataframe
            )

            assert 'No update results returned; no updates attempted' in caplog.text
            assert 'rows successfully updated' not in caplog.text
            assert 'updates failed.' not in caplog.text

    def test_update_existing_features_in_hosted_feature_layer_no_matching_rows_returns_0(self, mocker, caplog):
        pd_mock = mocker.Mock()
        pd_mock.return_value = pd.DataFrame({
            'objectId': [1, 2],
            'data': ['foo', 'bar'],
            'key': ['a', 'b'],
        })
        mocker.patch.object(pd.DataFrame.spatial, 'from_layer', new=pd_mock)
        class_mock = mocker.Mock()
        class_mock._class_logger = logging.getLogger('root')
        class_mock.new_dataframe = pd.DataFrame({
            'data': ['FOO', 'BAR'],
            'key': ['c', 'd'],
        })
        class_mock.index_column = 'key'

        updated_rows = palletjack.FeatureServiceInLineUpdater.update_existing_features_in_hosted_feature_layer(
            class_mock, '1234', ['data', 'key']
        )

        assert updated_rows == 0

    def test_update_existing_features_in_hosted_feature_layer_no_matching_rows_properly_logs(self, mocker, caplog):
        pd_mock = mocker.Mock()
        pd_mock.return_value = pd.DataFrame({
            'objectId': [1, 2],
            'data': ['foo', 'bar'],
            'key': ['a', 'b'],
        })
        mocker.patch.object(pd.DataFrame.spatial, 'from_layer', new=pd_mock)
        class_mock = mocker.Mock()
        class_mock._class_logger = logging.getLogger('root')
        class_mock.new_dataframe = pd.DataFrame({
            'data': ['FOO', 'BAR'],
            'key': ['c', 'd'],
        })
        class_mock.index_column = 'key'

        updated_rows = palletjack.FeatureServiceInLineUpdater.update_existing_features_in_hosted_feature_layer(
            class_mock, '1234', ['data', 'key']
        )

        assert 'No matching rows between live dataset and new dataset based on field `key`' in caplog.text


class TestSFTPLoader:

    def test_download_sftp_folder_contents_uses_right_credentials(self, mocker):
        sftploader_mock = mocker.Mock()
        sftploader_mock.knownhosts_file = 'knownhosts_file'
        sftploader_mock.host = 'sftp_host'
        sftploader_mock.username = 'username'
        sftploader_mock.password = 'password'
        download_dir_mock = mocker.Mock()
        download_dir_mock.iterdir.side_effect = [[], ['file_a', 'file_b']]
        sftploader_mock.download_dir = download_dir_mock

        connection_mock = mocker.MagicMock()
        context_manager_mock = mocker.MagicMock()
        context_manager_mock.return_value.__enter__.return_value = connection_mock
        mocker.patch('pysftp.Connection', new=context_manager_mock)

        cnopts_mock = mocker.Mock()
        cnopts_mock.side_effect = lambda knownhosts: knownhosts
        mocker.patch('pysftp.CnOpts', new=cnopts_mock)

        palletjack.SFTPLoader.download_sftp_folder_contents(sftploader_mock)

        context_manager_mock.assert_called_with(
            'sftp_host', username='username', password='password', cnopts='knownhosts_file'
        )

    def test_download_sftp_single_file_uses_right_credentials(self, mocker):
        sftploader_mock = mocker.Mock()
        sftploader_mock.knownhosts_file = 'knownhosts_file'
        sftploader_mock.host = 'sftp_host'
        sftploader_mock.username = 'username'
        sftploader_mock.password = 'password'
        sftploader_mock.download_dir = 'download_dir'

        connection_mock = mocker.MagicMock()
        context_manager_mock = mocker.MagicMock()
        context_manager_mock.return_value.__enter__.return_value = connection_mock
        mocker.patch('pysftp.Connection', new=context_manager_mock)

        cnopts_mock = mocker.Mock()
        cnopts_mock.side_effect = lambda knownhosts: knownhosts
        mocker.patch('pysftp.CnOpts', new=cnopts_mock)

        palletjack.SFTPLoader.download_sftp_single_file(sftploader_mock, 'filename', 'upload')

        context_manager_mock.assert_called_with(
            'sftp_host', username='username', password='password', cnopts='knownhosts_file', default_path='upload'
        )

    def test_read_csv_into_dataframe_with_column_names(self, mocker):
        pd_mock = mocker.Mock()
        pd_mock.return_value = pd.DataFrame()
        mocker.patch.object(pd, 'read_csv', new=pd_mock)

        sftploader_mock = mocker.Mock()
        sftploader_mock.download_dir = 'foo'

        column_types = {'bar': np.float64}

        palletjack.SFTPLoader.read_csv_into_dataframe(sftploader_mock, 'baz', column_types)

        pd_mock.assert_called_with(Path('foo', 'baz'), names=['bar'], dtype=column_types)

    def test_read_csv_into_dataframe_no_column_names(self, mocker):
        pd_mock = mocker.Mock()
        pd_mock.return_value = pd.DataFrame()
        mocker.patch.object(pd, 'read_csv', new=pd_mock)

        sftploader_mock = mocker.Mock()
        sftploader_mock.download_dir = 'foo'

        palletjack.SFTPLoader.read_csv_into_dataframe(sftploader_mock, 'baz')

        pd_mock.assert_called_with(Path('foo', 'baz'), names=None, dtype=None)


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
        reclassifier = palletjack.ColorRampReclassifier(webmap_item_mock, 'gis')

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
        reclassifier = palletjack.ColorRampReclassifier(webmap_item_mock, 'gis')

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
        reclassifier = palletjack.ColorRampReclassifier(webmap_item_mock, 'gis')

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
        reclassifier = palletjack.ColorRampReclassifier(webmap_item_mock, 'gis')

        with pytest.raises(ValueError) as error_info:
            layer_id = reclassifier._get_layer_id('foo')

        assert 'Could not find "foo" in test map' in str(error_info.value)

    def test_calculate_new_stops_with_manual_numbers(self):
        dataframe = pd.DataFrame({'numbers': [100, 300, 500, 700, 900]})

        stops = palletjack.ColorRampReclassifier._calculate_new_stops(dataframe, 'numbers', 5)

        assert stops == [100, 279, 458, 637, 816]

    def test_calculate_new_stops_mismatched_column_raises_error(self):
        dataframe = pd.DataFrame({'numbers': [100, 300, 500, 700, 900]})

        with pytest.raises(ValueError) as error_info:
            stops = palletjack.ColorRampReclassifier._calculate_new_stops(dataframe, 'foo', 5)
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
        reclassifier = palletjack.ColorRampReclassifier(webmap_item_mock, 'gis')

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
