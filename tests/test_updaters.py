import builtins
import json
import logging
import sys
import urllib
from pathlib import Path

import arcgis
import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest
import requests
from arcgis.features import GeoAccessor, GeoSeriesAccessor
from mock_arcpy import arcpy
from pandas.api.types import CategoricalDtype

import palletjack


@pytest.fixture
def combined_values():
    values = {
        1: {
            'old_values': {
                'objectId': 1,
                'data': 'foo',
            },
            'new_values': {
                'key': 'a',
                'data': 'FOO'
            }
        },
        2: {
            'old_values': {
                'objectId': 2,
                'data': 'bar',
            },
            'new_values': {
                'key': 'b',
                'data': 'BAR'
            }
        }
    }
    return values


@pytest.fixture
def hide_available_pkg(monkeypatch):
    """Mocks import to throw an ImportError (for error handling testing purposes) for a package that does, in fact, exist"""
    import_orig = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name == 'arcpy':
            raise ImportError()
        return import_orig(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', mocked_import)


class TestFeatureServiceInlineUpdater:

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
        fsupdater_mock.new_data_as_dict = {'12345': {'Count': '57', 'Amount': 100.00, 'Date': '1/1/2022'}}

        palletjack.FeatureServiceInlineUpdater.update_existing_features_in_feature_service_with_arcpy(
            fsupdater_mock, 'foo', ['ZipCode', 'Count', 'Amount', 'Date']
        )

        cursor_mock.updateRow.assert_called_with(['12345', '57', 100.00, '1/1/2022'])
        cursor_mock.updateRow.assert_called_once()

    @pytest.mark.usefixtures('hide_available_pkg')
    def test_update_existing_features_in_feature_service_with_arcpy_reports_error_on_import_failure(self, mocker):

        with pytest.raises(ImportError, match='Failure importing arcpy. ArcGIS Pro must be installed.'):
            palletjack.FeatureServiceInlineUpdater.update_existing_features_in_feature_service_with_arcpy(
                mocker.Mock(), 'foo', ['ZipCode', 'Count', 'Amount', 'Date']
            )

    def test_clean_dataframe_columns_renames_and_drops_columns(self, mocker):
        class_mock = mocker.Mock()
        fields = ['foo', 'bar']
        dataframe = pd.DataFrame(columns=['foo_x', 'bar_x', 'baz', 'foo_y', 'bar_y'])

        renamed = palletjack.FeatureServiceInlineUpdater._clean_dataframe_columns(class_mock, dataframe, fields)

        assert list(renamed.columns) == ['baz', 'foo', 'bar']

    def test_clean_dataframe_columns_handles_field_not_found_in_dataframe(self, mocker):
        class_mock = mocker.Mock()
        fields = ['foo', 'bar', 'buz']
        dataframe = pd.DataFrame(columns=['foo_x', 'bar_x', 'baz', 'foo_y', 'bar_y'])

        renamed = palletjack.FeatureServiceInlineUpdater._clean_dataframe_columns(class_mock, dataframe, fields)

        assert list(renamed.columns) == ['baz', 'foo', 'bar']

    def test_clean_dataframe_columns_deletes_merge_field(self, mocker):
        class_mock = mocker.Mock()
        fields = ['foo', 'bar', 'buz']
        dataframe = pd.DataFrame(columns=['foo_x', 'bar_x', 'baz', 'foo_y', 'bar_y', '_merge'])

        renamed = palletjack.FeatureServiceInlineUpdater._clean_dataframe_columns(class_mock, dataframe, fields)

        assert list(renamed.columns) == ['baz', 'foo', 'bar']

    def test_get_common_rows_joins_properly_all_rows_in_both(self, mocker):
        class_mock = mocker.Mock()
        class_mock.index_column = 'key'
        class_mock.new_dataframe = pd.DataFrame({'col1': [10, 20, 30], 'col2': [40, 50, 60], 'key': ['a', 'b', 'c']})
        live_dataframe = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'key': ['a', 'b', 'c']})

        joined = palletjack.FeatureServiceInlineUpdater._get_common_rows(class_mock, live_dataframe)

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

        joined = palletjack.FeatureServiceInlineUpdater._get_common_rows(class_mock, live_dataframe)

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

        joined = palletjack.FeatureServiceInlineUpdater._get_common_rows(class_mock, live_dataframe)

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


class TestFeatureServiceInlineUpdaterResultParsing:

    def test_parse_results_returns_correct_number_of_updated_rows(self, mocker, combined_values):
        class_mock = mocker.Mock()
        class_mock._get_old_and_new_values.return_value = (combined_values)
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

        rows_updated = palletjack.FeatureServiceInlineUpdater._parse_results(class_mock, results_dict, live_dataframe)

        assert rows_updated == 2

    def test_parse_results_returns_0_if_both_successful_and_failed_result(self, mocker, combined_values):
        class_mock = mocker.Mock()
        class_mock._get_old_and_new_values.return_value = (combined_values)
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

        rows_updated = palletjack.FeatureServiceInlineUpdater._parse_results(class_mock, results_dict, live_dataframe)

        assert rows_updated == 0

    def test_parse_results_logs_success_info(self, mocker, caplog, combined_values):
        class_mock = mocker.Mock()
        class_mock._class_logger = logging.getLogger('root')
        # combined_values = {
        #     1: {
        #         'old_values': {
        #             'objectId': 1,
        #             'data': 'foo',
        #         },
        #         'new_values': {
        #             'key': 'a',
        #             'data': 'FOO'
        #         }
        #     },
        #     2: {
        #         'old_values': {
        #             'objectId': 2,
        #             'data': 'bar',
        #         },
        #         'new_values': {
        #             'key': 'b',
        #             'data': 'BAR'
        #         }
        #     }
        # }
        class_mock._get_old_and_new_values.return_value = combined_values
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
            rows_updated = palletjack.FeatureServiceInlineUpdater._parse_results(
                class_mock, results_dict, live_dataframe
            )

            assert '2 rows successfully updated' in caplog.text
            assert "Existing data: {'objectId': 1, 'data': 'foo'}" not in caplog.text
            assert "New data: {'key': 'a', 'data': 'FOO'}" not in caplog.text
            assert "Existing data: {'objectId': 2, 'data': 'bar'}" not in caplog.text
            assert "New data: {'key': 'b', 'data': 'BAR'}" not in caplog.text

    def test_parse_results_logs_success_debug(self, mocker, caplog, combined_values):
        class_mock = mocker.Mock()
        class_mock._class_logger = logging.getLogger('root')
        # combined_values = {
        #     1: {
        #         'old_values': {
        #             'objectId': 1,
        #             'data': 'foo',
        #         },
        #         'new_values': {
        #             'key': 'a',
        #             'data': 'FOO'
        #         }
        #     },
        #     2: {
        #         'old_values': {
        #             'objectId': 2,
        #             'data': 'bar',
        #         },
        #         'new_values': {
        #             'key': 'b',
        #             'data': 'BAR'
        #         }
        #     }
        # }
        class_mock._get_old_and_new_values.return_value = combined_values
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
            rows_updated = palletjack.FeatureServiceInlineUpdater._parse_results(
                class_mock, results_dict, live_dataframe
            )

            assert '2 rows successfully updated' in caplog.text
            assert "Existing data: {'objectId': 1, 'data': 'foo'}" in caplog.text
            assert "New data: {'key': 'a', 'data': 'FOO'}" in caplog.text
            assert "Existing data: {'objectId': 2, 'data': 'bar'}" in caplog.text
            assert "New data: {'key': 'b', 'data': 'BAR'}" in caplog.text

    def test_parse_results_logs_failures_at_warning(self, mocker, caplog):
        class_mock = mocker.Mock()
        class_mock._class_logger = logging.getLogger('root')
        combined_values = [{
            1: {
                'old_values': {
                    'objectId': 1,
                    'data': 'foo',
                },
                'new_values': {
                    'key': 'a',
                    'data': 'FOO'
                }
            },
        }, {
            2: {
                'old_values': {
                    'objectId': 2,
                    'data': 'bar',
                },
                'new_values': {
                    'key': 'b',
                    'data': 'BAR'
                }
            }
        }]
        class_mock._get_old_and_new_values.side_effect = combined_values
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
            rows_updated = palletjack.FeatureServiceInlineUpdater._parse_results(
                class_mock, results_dict, live_dataframe
            )

            assert caplog.records[
                0
            ].message == 'The following 1 updates failed. As a result, all successful updates should have been rolled back.'
            assert caplog.records[0].levelname == 'WARNING'

            assert caplog.records[1].message == "Existing data: {'objectId': 2, 'data': 'bar'}"
            assert caplog.records[1].levelname == 'WARNING'

            assert caplog.records[2].message == "New data: {'key': 'b', 'data': 'BAR'}"
            assert caplog.records[2].levelname == 'WARNING'

    def test_parse_results_doesnt_log_any_success_on_all_failed(self, mocker, caplog):
        class_mock = mocker.Mock()
        class_mock._class_logger = logging.getLogger('root')
        combined_values = {
            1: {
                'old_values': {
                    'objectId': 1,
                    'data': 'foo',
                },
                'new_values': {
                    'key': 'a',
                    'data': 'FOO'
                }
            },
            2: {
                'old_values': {
                    'objectId': 2,
                    'data': 'bar',
                },
                'new_values': {
                    'key': 'b',
                    'data': 'BAR'
                }
            }
        }
        class_mock._get_old_and_new_values.return_value = combined_values
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
            rows_updated = palletjack.FeatureServiceInlineUpdater._parse_results(
                class_mock, results_dict, live_dataframe
            )

            assert 'rows successfully updated' not in caplog.text

    def test_parse_results_logs_successes_before_failure(self, mocker, caplog):
        class_mock = mocker.Mock()
        class_mock._class_logger = logging.getLogger('root')
        combined_values = [{
            1: {
                'old_values': {
                    'objectId': 1,
                    'data': 'foo',
                },
                'new_values': {
                    'key': 'a',
                    'data': 'FOO'
                }
            },
        }, {
            2: {
                'old_values': {
                    'objectId': 2,
                    'data': 'bar',
                },
                'new_values': {
                    'key': 'b',
                    'data': 'BAR'
                }
            }
        }]
        class_mock._get_old_and_new_values.side_effect = combined_values
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
            rows_updated = palletjack.FeatureServiceInlineUpdater._parse_results(
                class_mock, results_dict, live_dataframe
            )
            assert caplog.records[0].message == '1 rows successfully updated:'
            assert caplog.records[0].levelname == 'INFO'
            assert caplog.records[1].message == "Existing data: {'objectId': 1, 'data': 'foo'}"
            assert caplog.records[1].levelname == 'DEBUG'
            assert caplog.records[2].message == "New data: {'key': 'a', 'data': 'FOO'}"
            assert caplog.records[2].levelname == 'DEBUG'
            assert caplog.records[
                3
            ].message == 'The following 1 updates failed. As a result, all successful updates should have been rolled back.'
            assert caplog.records[3].levelname == 'WARNING'
            assert caplog.records[4].message == "Existing data: {'objectId': 2, 'data': 'bar'}"
            assert caplog.records[4].levelname == 'WARNING'
            assert caplog.records[5].message == "New data: {'key': 'b', 'data': 'BAR'}"
            assert caplog.records[5].levelname == 'WARNING'

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
            rows_updated = palletjack.FeatureServiceInlineUpdater._parse_results(
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
            rows_updated = palletjack.FeatureServiceInlineUpdater._parse_results(
                class_mock, results_dict, live_dataframe
            )

            assert 'No update results returned; no updates attempted' in caplog.text
            assert 'rows successfully updated' not in caplog.text
            assert 'updates failed.' not in caplog.text

    def test_get_old_and_new_values_with_correct_data(self, mocker):
        class_mock = mocker.Mock()
        class_mock.new_dataframe = pd.DataFrame.from_dict(
            orient='index',
            data={
                0: {
                    'foo': '42',
                    'bar': 32,
                    'key': 'a'
                },
                1: {
                    'foo': '56',
                    'bar': 8,
                    'key': 'b'
                },
            },
        )

        class_mock.index_column = 'key'
        live_dict = {
            1: {
                'OBJECTID': 2,
                'key': 'a',
                'foo': '42000',
                'bar': 32000
            },
            2: {
                'OBJECTID': 10,
                'key': 'b',
                'foo': '56000',
                'bar': 8000
            }
        }
        oids = [10, 2]

        combined_data = palletjack.FeatureServiceInlineUpdater._get_old_and_new_values(class_mock, live_dict, oids)

        assert combined_data == {
            2: {
                'old_values': {
                    'OBJECTID': 2,
                    'key': 'a',
                    'foo': '42000',
                    'bar': 32000
                },
                'new_values': {
                    'foo': '42',
                    'bar': 32,
                    'key': 'a',
                }
            },
            10: {
                'old_values': {
                    'OBJECTID': 10,
                    'key': 'b',
                    'foo': '56000',
                    'bar': 8000
                },
                'new_values': {
                    'foo': '56',
                    'bar': 8,
                    'key': 'b'
                }
            }
        }

    def test_get_old_and_new_values_only_includes_existing_data_that_match_passed_objectids(self, mocker):
        class_mock = mocker.Mock()
        class_mock.new_dataframe = pd.DataFrame.from_dict(
            orient='index',
            data={
                0: {
                    'foo': '42',
                    'bar': 32,
                    'key': 'a'
                },
                1: {
                    'foo': '56',
                    'bar': 8,
                    'key': 'b'
                },
            },
        )

        class_mock.index_column = 'key'
        live_dict = {
            1: {
                'OBJECTID': 2,
                'key': 'a',
                'foo': '42000',
                'bar': 32000
            },
            2: {
                'OBJECTID': 10,
                'key': 'b',
                'foo': '56000',
                'bar': 8000
            },
            3: {
                'OBJECTID': 42,
                'key': 'c',
                'foo': '-10',
                'bar': -88
            }
        }
        oids = [10, 2]

        combined_data = palletjack.FeatureServiceInlineUpdater._get_old_and_new_values(class_mock, live_dict, oids)

        assert combined_data == {
            2: {
                'old_values': {
                    'OBJECTID': 2,
                    'key': 'a',
                    'foo': '42000',
                    'bar': 32000
                },
                'new_values': {
                    'foo': '42',
                    'bar': 32,
                    'key': 'a',
                }
            },
            10: {
                'old_values': {
                    'OBJECTID': 10,
                    'key': 'b',
                    'foo': '56000',
                    'bar': 8000
                },
                'new_values': {
                    'foo': '56',
                    'bar': 8,
                    'key': 'b'
                }
            }
        }

    def test_get_old_and_new_values_only_returns_new_data_that_match_passed_objectids(self, mocker):
        class_mock = mocker.Mock()
        class_mock.new_dataframe = pd.DataFrame.from_dict(
            orient='index',
            data={
                0: {
                    'foo': '42',
                    'bar': 32,
                    'key': 'a'
                },
                1: {
                    'foo': '56',
                    'bar': 8,
                    'key': 'b'
                },
                2: {
                    'foo': '88',
                    'bar': 64,
                    'key': 'z'
                },
            },
        )

        class_mock.index_column = 'key'
        live_dict = {
            1: {
                'OBJECTID': 2,
                'key': 'a',
                'foo': '42000',
                'bar': 32000
            },
            2: {
                'OBJECTID': 10,
                'key': 'b',
                'foo': '56000',
                'bar': 8000
            }
        }
        oids = [10, 2]

        combined_data = palletjack.FeatureServiceInlineUpdater._get_old_and_new_values(class_mock, live_dict, oids)

        assert combined_data == {
            2: {
                'old_values': {
                    'OBJECTID': 2,
                    'key': 'a',
                    'foo': '42000',
                    'bar': 32000
                },
                'new_values': {
                    'foo': '42',
                    'bar': 32,
                    'key': 'a',
                }
            },
            10: {
                'old_values': {
                    'OBJECTID': 10,
                    'key': 'b',
                    'foo': '56000',
                    'bar': 8000
                },
                'new_values': {
                    'foo': '56',
                    'bar': 8,
                    'key': 'b'
                }
            }
        }


class TestFeatureServiceInlineUpdaterFieldValidation:

    def test_validate_working_fields_in_live_and_new_dataframes_doesnt_raise_when_matching(self, mocker):
        live_df = pd.DataFrame(columns=['field1', 'field2', 'field3'])
        new_df = pd.DataFrame(columns=['field1', 'field2', 'field3'])
        fields = ['field1', 'field2', 'field3']

        updater_mock = mocker.Mock()
        updater_mock.new_dataframe = new_df

        #: This shouldn't raise an exception and so the test should pass
        palletjack.FeatureServiceInlineUpdater._validate_working_fields_in_live_and_new_dataframes(
            updater_mock, live_df, fields
        )

    def test_validate_working_fields_in_live_and_new_dataframes_raises_not_in_live(self, mocker):
        live_df = pd.DataFrame(columns=['field1', 'field2'])
        new_df = pd.DataFrame(columns=['field1', 'field2', 'field3'])
        fields = ['field1', 'field2', 'field3']

        updater_mock = mocker.Mock()
        updater_mock.new_dataframe = new_df

        with pytest.raises(RuntimeError) as exc_info:
            palletjack.FeatureServiceInlineUpdater._validate_working_fields_in_live_and_new_dataframes(
                updater_mock, live_df, fields
            )
        assert exc_info.value.args[
            0
        ] == 'Field mismatch between defined fields and either new or live data.\nFields not in live data: {\'field3\'}\nFields not in new data: set()'

    def test_validate_working_fields_in_live_and_new_dataframes_raises_not_in_new(self, mocker):
        live_df = pd.DataFrame(columns=['field1', 'field2', 'field3'])
        new_df = pd.DataFrame(columns=['field1', 'field2'])
        fields = ['field1', 'field2', 'field3']

        updater_mock = mocker.Mock()
        updater_mock.new_dataframe = new_df

        with pytest.raises(RuntimeError) as exc_info:
            palletjack.FeatureServiceInlineUpdater._validate_working_fields_in_live_and_new_dataframes(
                updater_mock, live_df, fields
            )
        assert exc_info.value.args[
            0
        ] == 'Field mismatch between defined fields and either new or live data.\nFields not in live data: set()\nFields not in new data: {\'field3\'}'

    def test_validate_working_fields_in_live_and_new_dataframes_raises_not_in_both(self, mocker):
        live_df = pd.DataFrame(columns=['field1', 'field2'])
        new_df = pd.DataFrame(columns=['field1', 'field2'])
        fields = ['field1', 'field2', 'field3']

        updater_mock = mocker.Mock()
        updater_mock.new_dataframe = new_df

        with pytest.raises(RuntimeError) as exc_info:
            palletjack.FeatureServiceInlineUpdater._validate_working_fields_in_live_and_new_dataframes(
                updater_mock, live_df, fields
            )
        assert exc_info.value.args[
            0
        ] == 'Field mismatch between defined fields and either new or live data.\nFields not in live data: {\'field3\'}\nFields not in new data: {\'field3\'}'


class TestFeatureServiceInlineUpdaterIntegrated:

    def test_update_existing_features_in_hosted_feature_layer_no_matching_rows_returns_0(self, mocker, caplog):
        pd_mock = mocker.Mock()
        pd_mock.return_value = pd.DataFrame({
            'OBJECTID': [1, 2],
            'data': ['foo', 'bar'],
            'key': ['a', 'b'],
        })
        mocker.patch.object(pd.DataFrame.spatial, 'from_layer', new=pd_mock)
        mocker.patch.object(arcgis.features.FeatureLayer, 'fromitem')

        new_dataframe = pd.DataFrame({
            'data': ['FOO', 'BAR'],
            'key': ['c', 'd'],
        })
        gis_mock = mocker.Mock()
        updater = palletjack.FeatureServiceInlineUpdater(gis_mock, new_dataframe, 'key')

        updated_rows = updater.update_existing_features_in_hosted_feature_layer('1234', ['data', 'key'])

        assert updated_rows == 0

    def test_update_existing_features_in_hosted_feature_layer_no_matching_rows_properly_logs(self, mocker, caplog):
        pd_mock = mocker.Mock()
        pd_mock.return_value = pd.DataFrame({
            'OBJECTID': [1, 2],
            'data': ['foo', 'bar'],
            'key': ['a', 'b'],
        })
        mocker.patch.object(pd.DataFrame.spatial, 'from_layer', new=pd_mock)
        mocker.patch.object(arcgis.features.FeatureLayer, 'fromitem')

        new_dataframe = pd.DataFrame({
            'data': ['FOO', 'BAR'],
            'key': ['c', 'd'],
        })
        gis_mock = mocker.Mock()
        updater = palletjack.FeatureServiceInlineUpdater(gis_mock, new_dataframe, 'key')

        updated_rows = updater.update_existing_features_in_hosted_feature_layer('1234', ['data', 'key'])

        assert 'No matching rows between live dataset and new dataset based on field `key`' in caplog.text
        assert "The following keys from the new data were not found in the existing dataset: ['c', 'd']" in caplog.text

    def test_update_existing_features_in_hosted_feature_layer_all_matching_returns_2(self, mocker, caplog):
        pd_mock = mocker.Mock()
        pd_mock.return_value = pd.DataFrame({
            'OBJECTID': [1, 2],
            'data': ['foo', 'bar'],
            'key': ['a', 'b'],
        })
        mocker.patch.object(pd.DataFrame.spatial, 'from_layer', new=pd_mock)

        new_dataframe = pd.DataFrame({
            'data': ['FOO', 'BAR'],
            'key': ['a', 'b'],
        })
        gis_mock = mocker.Mock()
        fromitem_function_mock = mocker.Mock()
        fromitem_function_mock.return_value.edit_features.return_value = {
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
        mocker.patch.object(arcgis.features.FeatureLayer, 'fromitem', new=fromitem_function_mock)
        updater = palletjack.FeatureServiceInlineUpdater(gis_mock, new_dataframe, 'key')

        updated_rows = updater.update_existing_features_in_hosted_feature_layer('1234', ['data', 'key'])

        assert updated_rows == 2

    def test_update_existing_features_in_hosted_feature_layer_all_matching_logs_properly(self, mocker, caplog):
        pd_mock = mocker.Mock()
        pd_mock.return_value = pd.DataFrame({
            'OBJECTID': [1, 2],
            'data': ['foo', 'bar'],
            'key': ['a', 'b'],
        })
        mocker.patch.object(pd.DataFrame.spatial, 'from_layer', new=pd_mock)
        new_dataframe = pd.DataFrame({
            'data': ['FOO', 'BAR'],
            'key': ['a', 'b'],
        })
        gis_mock = mocker.Mock()
        fromitem_function_mock = mocker.Mock()
        fromitem_function_mock.return_value.edit_features.return_value = {
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
        mocker.patch.object(arcgis.features.FeatureLayer, 'fromitem', new=fromitem_function_mock)
        updater = palletjack.FeatureServiceInlineUpdater(gis_mock, new_dataframe, 'key')

        with caplog.at_level(logging.DEBUG):
            updated_rows = updater.update_existing_features_in_hosted_feature_layer('1234', ['data', 'key'])
            assert '2 rows successfully updated:' in caplog.text
            assert "Existing data: {'OBJECTID': 1, 'data': 'foo', 'key': 'a'}" in caplog.text
            assert "New data: {'data': 'FOO', 'key': 'a'}" in caplog.text
            assert "Existing data: {'OBJECTID': 2, 'data': 'bar', 'key': 'b'}" in caplog.text
            assert "New data: {'data': 'BAR', 'key': 'b'}" in caplog.text

    def test_update_existing_features_in_hosted_feature_layer_subsets_logging_fields_properly(self, mocker, caplog):
        pd_mock = mocker.Mock()
        pd_mock.return_value = pd.DataFrame({
            'OBJECTID': [1, 2],
            'data': ['foo', 'bar'],
            'key': ['a', 'b'],
            'extra_field': ['muck', 'slime']
        })
        mocker.patch.object(pd.DataFrame.spatial, 'from_layer', new=pd_mock)
        new_dataframe = pd.DataFrame({
            'data': ['FOO', 'BAR'],
            'key': ['a', 'b'],
        })
        gis_mock = mocker.Mock()
        fromitem_function_mock = mocker.Mock()
        fromitem_function_mock.return_value.edit_features.return_value = {
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
        mocker.patch.object(arcgis.features.FeatureLayer, 'fromitem', new=fromitem_function_mock)
        updater = palletjack.FeatureServiceInlineUpdater(gis_mock, new_dataframe, 'key')

        with caplog.at_level(logging.DEBUG):
            updated_rows = updater.update_existing_features_in_hosted_feature_layer('1234', ['data', 'key'])
            assert '2 rows successfully updated:' in caplog.text
            assert "Existing data: {'OBJECTID': 1, 'data': 'foo', 'key': 'a'}" in caplog.text
            assert "New data: {'data': 'FOO', 'key': 'a'}" in caplog.text
            assert "Existing data: {'OBJECTID': 2, 'data': 'bar', 'key': 'b'}" in caplog.text
            assert "New data: {'data': 'BAR', 'key': 'b'}" in caplog.text


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


class TestAttachments:

    def test_create_attachment_action_df_adds_for_blank_existing_name(self, mocker):
        input_df = pd.DataFrame({
            'NAME': [np.nan],
            'new_path': ['bee/foo.png'],
        })

        ops_df = palletjack.FeatureServiceAttachmentsUpdater._create_attachment_action_df(
            mocker.Mock(), input_df, 'new_path'
        )

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

        ops_df = palletjack.FeatureServiceAttachmentsUpdater._create_attachment_action_df(
            mocker.Mock(), input_df, 'new_path'
        )

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

        ops_df = palletjack.FeatureServiceAttachmentsUpdater._create_attachment_action_df(
            mocker.Mock(), input_df, 'new_path'
        )

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

        ops_df = palletjack.FeatureServiceAttachmentsUpdater._create_attachment_action_df(
            mocker.Mock(), input_df, 'new_path'
        )

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

        ops_df = palletjack.FeatureServiceAttachmentsUpdater._create_attachment_action_df(
            mocker.Mock(), input_df, 'new_path'
        )

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

        live_data_subset = palletjack.FeatureServiceAttachmentsUpdater._get_live_oid_and_guid_from_join_field_values(
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

        current_attachments_df = palletjack.FeatureServiceAttachmentsUpdater._get_current_attachment_info_by_oid(
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
        palletjack.FeatureServiceAttachmentsUpdater._check_attachment_dataframe_for_invalid_column_names(
            dataframe, invalid_names
        )

    def test_check_attachment_dataframe_for_invalid_column_names_raises_with_one_invalid(self, mocker):
        dataframe = pd.DataFrame(columns=['foo', 'bar'])
        invalid_names = ['foo', 'boo']
        with pytest.raises(RuntimeError) as exc_info:
            palletjack.FeatureServiceAttachmentsUpdater._check_attachment_dataframe_for_invalid_column_names(
                dataframe, invalid_names
            )
        assert exc_info.value.args[0] == 'Attachment dataframe contains the following invalid names: [\'foo\']'

    def test_check_attachment_dataframe_for_invalid_column_names_raises_with_all_invalid(self, mocker):
        dataframe = pd.DataFrame(columns=['foo', 'bar'])
        invalid_names = ['foo', 'bar']
        with pytest.raises(RuntimeError) as exc_info:
            palletjack.FeatureServiceAttachmentsUpdater._check_attachment_dataframe_for_invalid_column_names(
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
            count = palletjack.FeatureServiceAttachmentsUpdater._add_attachments_by_oid(updater_mock, action_df, 'path')

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

        updater = palletjack.FeatureServiceAttachmentsUpdater(mocker.Mock())
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

        updater = palletjack.FeatureServiceAttachmentsUpdater(mocker.Mock())
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

        count = palletjack.FeatureServiceAttachmentsUpdater._add_attachments_by_oid(updater_mock, action_df, 'path')

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
            count = palletjack.FeatureServiceAttachmentsUpdater._overwrite_attachments_by_oid(
                updater_mock, action_df, 'path'
            )

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

        updater = palletjack.FeatureServiceAttachmentsUpdater(mocker.Mock())
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

        updater = palletjack.FeatureServiceAttachmentsUpdater(mocker.Mock())
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

        count = palletjack.FeatureServiceAttachmentsUpdater._overwrite_attachments_by_oid(
            updater_mock, action_df, 'path'
        )

        assert updater_mock.feature_layer.attachments.update.call_count == 1
        assert count == 1

    def test_create_attachments_dataframe_subsets_and_crafts_paths_properly(self, mocker):
        input_df = pd.DataFrame({
            'join': [1, 2, 3],
            'pic': ['foo.png', 'bar.png', 'baz.png'],
            'data': [11., 12., 13.],
        })

        attachment_df = palletjack.FeatureServiceAttachmentsUpdater.build_attachments_dataframe(
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

        attachment_df = palletjack.FeatureServiceAttachmentsUpdater.build_attachments_dataframe(
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

    def test_rename_columns_from_agol_handles_special_and_space(self):
        cols = ['Test Name:']

        renamed = palletjack.FeatureServiceOverwriter._rename_columns_for_agol(cols)

        assert renamed == {'Test Name:': 'Test_Name_'}

    def test_rename_columns_from_agol_handles_special_and_space_leaves_others_alone(self):
        cols = ['Test Name:', 'FooName']

        renamed = palletjack.FeatureServiceOverwriter._rename_columns_for_agol(cols)

        assert renamed == {'Test Name:': 'Test_Name_', 'FooName': 'FooName'}

    def test_rename_columns_from_agol_retains_underscores(self):
        cols = ['Test Name:', 'Foo_Name']

        renamed = palletjack.FeatureServiceOverwriter._rename_columns_for_agol(cols)

        assert renamed == {'Test Name:': 'Test_Name_', 'Foo_Name': 'Foo_Name'}

    def test_check_fields_match_normal(self, mocker):
        mock_fl = mocker.Mock()
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
        df = pd.DataFrame(columns=['Foo', 'Bar'])
        overwriter = palletjack.FeatureServiceOverwriter(mocker.Mock())

        palletjack.FeatureServiceOverwriter._check_fields_match(overwriter, mock_fl, df)

    def test_check_fields_match_raises_error_on_extra_new_field(self, mocker):
        mock_fl = mocker.Mock()
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
        df = pd.DataFrame(columns=['Foo', 'Bar', 'Baz'])
        overwriter = palletjack.FeatureServiceOverwriter(mocker.Mock())

        with pytest.raises(RuntimeError) as exc_info:
            palletjack.FeatureServiceOverwriter._check_fields_match(overwriter, mock_fl, df)

        assert exc_info.value.args[
            0] == 'New dataset contains the following fields that are not present in the live dataset: {\'Baz\'}'

    def test_check_fields_match_ignores_new_shape_field(self, mocker):
        mock_fl = mocker.Mock()
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
        df = pd.DataFrame(columns=['Foo', 'Bar', 'SHAPE'])
        overwriter = palletjack.FeatureServiceOverwriter(mocker.Mock())

        palletjack.FeatureServiceOverwriter._check_fields_match(overwriter, mock_fl, df)

    def test_check_fields_match_warns_on_missing_new_field(self, mocker, caplog):
        mock_fl = mocker.Mock()
        mock_fl.properties = {
            'fields': [
                {
                    'name': 'Foo'
                },
                {
                    'name': 'Bar'
                },
                {
                    'name': 'Baz'
                },
            ]
        }
        df = pd.DataFrame(columns=['Foo', 'Bar'])
        overwriter = palletjack.FeatureServiceOverwriter(mocker.Mock())

        palletjack.FeatureServiceOverwriter._check_fields_match(overwriter, mock_fl, df)

        assert 'New dataset does not contain the following fields that are present in the live dataset: {\'Baz\'}' in caplog.text

    def test_truncate_existing_data_normal(self, mocker):
        fl_mock = mocker.Mock()
        fl_mock.manager.truncate.return_value = {
            'submissionTime': 123,
            'lastUpdatedTime': 124,
            'status': 'Completed',
        }
        overwriter = palletjack.FeatureServiceOverwriter(mocker.Mock())

        palletjack.FeatureServiceOverwriter._truncate_existing_data(overwriter, fl_mock, 0, 'abc')

    def test_truncate_existing_raises_error_on_failure(self, mocker):
        fl_mock = mocker.Mock()
        fl_mock.manager.truncate.return_value = {
            'submissionTime': 123,
            'lastUpdatedTime': 124,
            'status': 'Foo',
        }
        overwriter = palletjack.FeatureServiceOverwriter(mocker.Mock())

        with pytest.raises(RuntimeError) as exc_info:
            palletjack.FeatureServiceOverwriter._truncate_existing_data(overwriter, fl_mock, 0, 'abc')

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
        overwriter = palletjack.FeatureServiceOverwriter(mocker.Mock())

        palletjack.FeatureServiceOverwriter._truncate_existing_data(overwriter, fl_mock, 0, 'abc')

    def test_retry_returns_on_first_success(self, mocker):
        mock = mocker.Mock()
        mock.function.return_value = 42

        overwriter = palletjack.FeatureServiceOverwriter(mocker.Mock())

        answer = overwriter._retry(mock.function, 'a', 'b')

        assert answer == 42
        assert mock.function.call_count == 1

    def test_retry_returns_after_one_failure(self, mocker):
        mock = mocker.Mock()
        mock.function.side_effect = [urllib.error.HTTPError('a', 'b', 'c', 'd', 'e'), 42]
        mocker.patch('palletjack.updaters.sleep')

        overwriter = palletjack.FeatureServiceOverwriter(mocker.Mock())

        answer = overwriter._retry(mock.function, 'a', 'b')

        assert answer == 42
        assert mock.function.call_count == 2

    def test_retry_returns_after_two_failures(self, mocker):
        mock = mocker.Mock()
        mock.function.side_effect = [
            urllib.error.HTTPError('a', 'b', 'c', 'd', 'e'),
            urllib.error.HTTPError('a', 'b', 'c', 'd', 'e'),
            42,
        ]
        mocker.patch('palletjack.updaters.sleep')

        overwriter = palletjack.FeatureServiceOverwriter(mocker.Mock())

        answer = overwriter._retry(mock.function, 'a', 'b')

        assert answer == 42
        assert mock.function.call_count == 3

    def test_retry_fails_after_four_failures(self, mocker):
        mock = mocker.Mock()
        mock.function.side_effect = [
            urllib.error.HTTPError('a', 'b', 'c', 'd', 'e'),
            urllib.error.HTTPError('a', 'b', 'c', 'd', 'e'),
            urllib.error.HTTPError('a', 'b', 'c', 'd', 'e'),
            urllib.error.HTTPError('a', 'b', 'c', 'd', 'e'),
            42,
        ]
        mocker.patch('palletjack.updaters.sleep')

        overwriter = palletjack.FeatureServiceOverwriter(mocker.Mock())

        with pytest.raises(urllib.error.HTTPError):
            answer = overwriter._retry(mock.function, 'a', 'b')

        # assert answer == 42
        assert mock.function.call_count == 4

    def test_retry_retries_on_request_http_error(self, mocker):
        mock = mocker.Mock()
        mock.function.side_effect = [requests.exceptions.HTTPError, 42]
        mocker.patch('palletjack.updaters.sleep')

        overwriter = palletjack.FeatureServiceOverwriter(mocker.Mock())

        answer = overwriter._retry(mock.function, 'a', 'b')

        assert answer == 42

    def test_retry_retries_on_request_ssl_error(self, mocker):
        mock = mocker.Mock()
        mock.function.side_effect = [requests.exceptions.SSLError, 42]
        mocker.patch('palletjack.updaters.sleep')

        overwriter = palletjack.FeatureServiceOverwriter(mocker.Mock())

        answer = overwriter._retry(mock.function, 'a', 'b')

        assert answer == 42

    def test_append_new_data_doesnt_raise_on_normal(self, mocker):
        mock_df = mocker.Mock()
        mock_fl = mocker.Mock()
        mock_fl.append.return_value = (True, {'message': 'foo'})

        overwriter = palletjack.FeatureServiceOverwriter(mocker.Mock())
        overwriter._append_new_data(mock_fl, mock_df, 0, 'abc')

    def test_append_new_data_retries_on_httperror(self, mocker):
        mock_df = mocker.Mock()
        mock_fl = mocker.Mock()
        mock_fl.append.side_effect = [urllib.error.HTTPError('a', 'b', 'c', 'd', 'e'), (True, {'message': 'foo'})]

        overwriter = palletjack.FeatureServiceOverwriter(mocker.Mock())
        overwriter._append_new_data(mock_fl, mock_df, 0, 'abc')

    def test_append_new_data_raises_on_False_result(self, mocker):
        mock_df = mocker.Mock()
        mock_fl = mocker.Mock()
        mock_fl.append.return_value = (False, {'message': 'foo'})

        overwriter = palletjack.FeatureServiceOverwriter(mocker.Mock())

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

        overwriter = palletjack.FeatureServiceOverwriter(mocker.Mock())

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

        overwriter = palletjack.FeatureServiceOverwriter(mocker.Mock())

        with pytest.raises(
            RuntimeError,
            match='Failed to append data to layer id 0 in itemid abc. Append should have been rolled back.'
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

        mocker.patch.object(palletjack.FeatureServiceOverwriter, '_save_truncated_data', return_value='/foo/bar.json')

        overwriter = palletjack.FeatureServiceOverwriter(mocker.Mock())

        with pytest.raises(RuntimeError) as exc_info:
            uploaded_features = overwriter.truncate_and_load_feature_service('abc', new_dataframe, 'foo/dir')

        assert exc_info.value.args[
            0] == 'Failed to re-add truncated data after failed append; data saved to /foo/bar.json'
        assert 'Append failed; attempting to re-load truncated data...' in caplog.text
        assert 'features reloaded' not in caplog.text

    def test_replace_nan_series_with_empty_strings_one_empty_one_non_empty_float(self, mocker):
        df = pd.DataFrame({
            'normal': [1., 2., 3.],
            'empty': [np.nan, np.nan, np.nan],
        })

        fixed_df = palletjack.FeatureServiceOverwriter._replace_nan_series_with_empty_strings(mocker.Mock(), df)

        test_df = pd.DataFrame({
            'normal': [1., 2., 3.],
            'empty': ['', '', ''],
        })

        tm.assert_frame_equal(fixed_df, test_df)

    def test_replace_nan_series_with_empty_strings_other_series_has_nan(self, mocker):
        df = pd.DataFrame({
            'normal': [1., 2., np.nan],
            'empty': [np.nan, np.nan, np.nan],
        })

        fixed_df = palletjack.FeatureServiceOverwriter._replace_nan_series_with_empty_strings(mocker.Mock(), df)

        test_df = pd.DataFrame({
            'normal': [1., 2., np.nan],
            'empty': ['', '', ''],
        })

        tm.assert_frame_equal(fixed_df, test_df)

    def test_replace_nan_series_with_empty_strings_other_series_is_empty_str(self, mocker):
        df = pd.DataFrame({
            'normal': ['', '', ''],
            'empty': [np.nan, np.nan, np.nan],
        })

        fixed_df = palletjack.FeatureServiceOverwriter._replace_nan_series_with_empty_strings(mocker.Mock(), df)

        test_df = pd.DataFrame({
            'normal': ['', '', ''],
            'empty': ['', '', ''],
        })

        tm.assert_frame_equal(fixed_df, test_df)

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

        palletjack.FeatureServiceOverwriter._save_truncated_data(mocker.Mock(), mock_sdf, 'foo')

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
        mocker.patch('palletjack.updaters.datetime', new=datetime_mock)

        open_mock = mocker.MagicMock()
        context_manager_mock = mocker.MagicMock()
        context_manager_mock.return_value.__enter__.return_value = open_mock
        mocker.patch('pathlib.Path.open', new=context_manager_mock)

        out_path = palletjack.FeatureServiceOverwriter._save_truncated_data(mocker.Mock(), mock_sdf, 'foo')

        assert out_path == Path('foo/old_data_foo-date.json')
