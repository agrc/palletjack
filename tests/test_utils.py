import logging

import numpy as np
import pandas as pd
import pytest
from pandas import testing as tm

import palletjack


class TestRenameColumns:

    def test_rename_columns_from_agol_handles_special_and_space(self):
        cols = ['Test Name:']

        renamed = palletjack.utils.rename_columns_for_agol(cols)

        assert renamed == {'Test Name:': 'Test_Name_'}

    def test_rename_columns_from_agol_handles_special_and_space_leaves_others_alone(self):
        cols = ['Test Name:', 'FooName']

        renamed = palletjack.utils.rename_columns_for_agol(cols)

        assert renamed == {'Test Name:': 'Test_Name_', 'FooName': 'FooName'}

    def test_rename_columns_from_agol_retains_underscores(self):
        cols = ['Test Name:', 'Foo_Name']

        renamed = palletjack.utils.rename_columns_for_agol(cols)

        assert renamed == {'Test Name:': 'Test_Name_', 'Foo_Name': 'Foo_Name'}


class TestCheckFieldsMatch:

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

        palletjack.utils.check_fields_match(mock_fl, df)

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

        with pytest.raises(RuntimeError) as exc_info:
            palletjack.utils.check_fields_match(mock_fl, df)

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

        palletjack.utils.check_fields_match(mock_fl, df)

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

        palletjack.utils.check_fields_match(mock_fl, df)

        assert 'New dataset does not contain the following fields that are present in the live dataset: {\'Baz\'}' in caplog.text


class TestRetry:

    def test_retry_returns_on_first_success(self, mocker):
        mock = mocker.Mock()
        mock.function.return_value = 42

        answer = palletjack.utils.retry(mock.function, 'a', 'b')

        assert answer == 42
        assert mock.function.call_count == 1

    def test_retry_returns_after_one_failure(self, mocker):
        mock = mocker.Mock()
        mock.function.side_effect = [Exception, 42]
        mocker.patch('palletjack.utils.sleep')

        answer = palletjack.utils.retry(mock.function, 'a', 'b')

        assert answer == 42
        assert mock.function.call_count == 2

    def test_retry_returns_after_two_failures(self, mocker):
        mock = mocker.Mock()
        mock.function.side_effect = [
            Exception,
            Exception,
            42,
        ]
        mocker.patch('palletjack.utils.sleep')

        answer = palletjack.utils.retry(mock.function, 'a', 'b')

        assert answer == 42
        assert mock.function.call_count == 3

    def test_retry_fails_after_four_failures(self, mocker):
        mock = mocker.Mock()
        mock.function.side_effect = [
            Exception,
            Exception,
            Exception,
            Exception,
            42,
        ]
        mocker.patch('palletjack.utils.sleep')

        with pytest.raises(Exception):
            answer = palletjack.utils.retry(mock.function, 'a', 'b')

        # assert answer == 42
        assert mock.function.call_count == 4


class TestReplaceNanSeries:

    def test_replace_nan_series_with_empty_strings_one_empty_one_non_empty_float(self, mocker):
        df = pd.DataFrame({
            'normal': [1., 2., 3.],
            'empty': [np.nan, np.nan, np.nan],
        })

        fixed_df = palletjack.utils.replace_nan_series_with_empty_strings(df)

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

        fixed_df = palletjack.utils.replace_nan_series_with_empty_strings(df)

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

        fixed_df = palletjack.utils.replace_nan_series_with_empty_strings(df)

        test_df = pd.DataFrame({
            'normal': ['', '', ''],
            'empty': ['', '', ''],
        })

        tm.assert_frame_equal(fixed_df, test_df)


class TestCheckIndexColumnInFL:

    def test_check_index_column_in_feature_layer_doesnt_raise_on_normal(self, mocker):
        fl_mock = mocker.Mock()
        fl_mock.properties = {
            'fields': [
                {
                    'name': 'Foo'
                },
                {
                    'name': 'Bar'
                },
            ]
        }

        palletjack.utils.check_index_column_in_feature_layer(fl_mock, 'Foo')

    def test_check_index_column_in_feature_layer_raises_on_missing(self, mocker):
        fl_mock = mocker.Mock()
        fl_mock.properties = {
            'fields': [
                {
                    'name': 'Foo'
                },
                {
                    'name': 'Bar'
                },
            ]
        }

        with pytest.raises(RuntimeError) as exc_info:
            palletjack.utils.check_index_column_in_feature_layer(fl_mock, 'Baz')

        assert exc_info.value.args[0] == 'Index column Baz not found in feature layer fields [\'Foo\', \'Bar\']'


class TestCheckFieldUnique:

    def test_check_field_set_to_unique_doesnt_raise_on_normal(self, mocker):
        mock_fl = mocker.Mock()
        mock_fl.properties = {
            'indexes': [
                {
                    'fields': 'foo_field',
                    'isUnique': True
                },
            ]
        }

        palletjack.utils.check_field_set_to_unique(mock_fl, 'foo_field')

    def test_check_field_set_to_unique_raises_on_non_indexed_field(self, mocker):
        mock_fl = mocker.Mock()
        mock_fl.properties = {
            'indexes': [
                {
                    'fields': 'bar_field',
                    'isUnique': True
                },
            ]
        }

        with pytest.raises(RuntimeError) as exc_info:
            palletjack.utils.check_field_set_to_unique(mock_fl, 'foo_field')

        assert exc_info.value.args[0] == 'foo_field does not have a "unique constraint" set within the feature layer'

    def test_check_field_set_to_unique_raises_on_non_unique_field(self, mocker):
        mock_fl = mocker.Mock()
        mock_fl.properties = {
            'indexes': [
                {
                    'fields': 'foo_field',
                    'isUnique': False
                },
            ]
        }

        with pytest.raises(RuntimeError) as exc_info:
            palletjack.utils.check_field_set_to_unique(mock_fl, 'foo_field')

        assert exc_info.value.args[0] == 'foo_field does not have a "unique constraint" set within the feature layer'


class TestGeocodeAddr:

    def test_geocode_addr_builds_url_correctly(self, mocker):
        mocker.patch('palletjack.utils.requests', autospec=True)
        mocker.patch('palletjack.utils.sleep')

        response_mock = mocker.Mock()
        response_mock.json.return_value = {
            'status': 200,
            'result': {
                'location': {
                    'x': 123,
                    'y': 456
                },
                'score': 100.,
                'matchAddress': 'bar'
            }
        }
        response_mock.status_code = 200

        palletjack.utils.requests.get.return_value = response_mock

        palletjack.utils.geocode_addr('foo', 'bar', 'foo_key', (0.015, 0.03))

        palletjack.utils.requests.get.assert_called_with(
            'https://api.mapserv.utah.gov/api/v1/geocode/foo/bar', params={'apiKey': 'foo_key'}
        )

    def test_geocode_addr_handles_kwargs_for_geocoding_api(self, mocker):
        mocker.patch('palletjack.utils.requests', autospec=True)
        mocker.patch('palletjack.utils.sleep')

        response_mock = mocker.Mock()
        response_mock.json.return_value = {
            'status': 200,
            'result': {
                'location': {
                    'x': 123,
                    'y': 456
                },
                'score': 100.,
                'matchAddress': 'bar'
            }
        }
        response_mock.status_code = 200

        palletjack.utils.requests.get.return_value = response_mock

        palletjack.utils.geocode_addr('foo', 'bar', 'foo_key', (0.015, 0.03), spatialReference=3857)

        palletjack.utils.requests.get.assert_called_with(
            'https://api.mapserv.utah.gov/api/v1/geocode/foo/bar',
            params={
                'apiKey': 'foo_key',
                'spatialReference': 3857
            }
        )

    def test_geocode_addr_returns_coords(self, mocker):
        mocker.patch('palletjack.utils.requests', autospec=True)
        mocker.patch('palletjack.utils.sleep')

        response_mock = mocker.Mock()
        response_mock.json.return_value = {
            'status': 200,
            'result': {
                'location': {
                    'x': 123,
                    'y': 456
                },
                'score': 100.,
                'matchAddress': 'bar'
            }
        }
        response_mock.status_code = 200

        palletjack.utils.requests.get.return_value = response_mock

        result = palletjack.utils.geocode_addr('foo', 'bar', 'foo_key', (0.015, 0.03))

        assert result == (123, 456, 100., 'bar')

    def test_geocode_addr_returns_null_island_bad_status_code(self, mocker):
        mocker.patch('palletjack.utils.requests', autospec=True)
        mocker.patch('palletjack.utils.sleep')

        response_mock = mocker.Mock()
        response_mock.status_code = 404

        palletjack.utils.requests.get.return_value = response_mock

        result = palletjack.utils.geocode_addr('foo', 'bar', 'foo_key', (0.015, 0.03))

        assert result == (0, 0, 0., 'No Match')

    def test_geocode_addr_returns_null_island_bad_status_return_value(self, mocker):
        mocker.patch('palletjack.utils.requests', autospec=True)
        mocker.patch('palletjack.utils.sleep')

        response_mock = mocker.Mock()
        response_mock.status_code = 404

        palletjack.utils.requests.get.return_value = response_mock

        result = palletjack.utils.geocode_addr('foo', 'bar', 'foo_key', (0.015, 0.03))

        assert result == (0, 0, 0., 'No Match')

    def test_geocode_addr_retries_on_exception(self, mocker):
        mocker.patch('palletjack.utils.requests', autospec=True)
        mocker.patch('palletjack.utils.sleep')

        response_mock = mocker.Mock()
        response_mock.json.return_value = {
            'status': 200,
            'result': {
                'location': {
                    'x': 123,
                    'y': 456
                },
                'score': 100.,
                'matchAddress': 'bar'
            }
        }
        response_mock.status_code = 200

        palletjack.utils.requests.get.side_effect = [Exception, response_mock]

        result = palletjack.utils.geocode_addr('foo', 'bar', 'foo_key', (0.015, 0.03))

        assert palletjack.utils.requests.get.call_count == 2
        assert result == (123, 456, 100., 'bar')

    def test_geocode_addr_retries_on_missing_response(self, mocker, caplog):
        caplog.set_level(logging.DEBUG)
        mocker.patch('palletjack.utils.requests', autospec=True)
        mocker.patch('palletjack.utils.sleep')

        response_mock = mocker.Mock()
        response_mock.json.return_value = {
            'status': 200,
            'result': {
                'location': {
                    'x': 123,
                    'y': 456
                },
                'score': 100.,
                'matchAddress': 'bar'
            }
        }
        response_mock.status_code = 200

        palletjack.utils.requests.get.side_effect = [None, response_mock]

        result = palletjack.utils.geocode_addr('foo', 'bar', 'foo_key', (0.015, 0.03))

        assert 'No response from GET; server nodejs timeout?' in caplog.text
        assert palletjack.utils.requests.get.call_count == 2
        assert result == (123, 456, 100., 'bar')

    def test_geocode_addr_retries_on_bad_status_code(self, mocker, caplog):
        caplog.set_level(logging.DEBUG)
        mocker.patch('palletjack.utils.requests', autospec=True)
        mocker.patch('palletjack.utils.sleep')

        first_response_mock = mocker.Mock()
        first_response_mock.status_code = 500

        second_response_mock = mocker.Mock()
        second_response_mock.json.return_value = {
            'status': 200,
            'result': {
                'location': {
                    'x': 123,
                    'y': 456
                },
                'score': 100.,
                'matchAddress': 'bar'
            }
        }
        second_response_mock.status_code = 200

        palletjack.utils.requests.get.side_effect = [first_response_mock, second_response_mock]

        result = palletjack.utils.geocode_addr('foo', 'bar', 'foo_key', (0.015, 0.03))

        assert 'Did not receive a valid geocoding response; status code: 500' in caplog.text
        assert palletjack.utils.requests.get.call_count == 2
        assert result == (123, 456, 100., 'bar')

    def test_geocode_addr_handles_complete_geocode_failure(self, mocker, caplog):
        caplog.set_level(logging.DEBUG)
        mocker.patch('palletjack.utils.requests', autospec=True)
        mocker.patch('palletjack.utils.sleep')

        bad_response = mocker.Mock()
        bad_response.status_code = 500
        palletjack.utils.requests.get.side_effect = [bad_response] * 4

        result = palletjack.utils.geocode_addr('foo', 'bar', 'foo_key', (0.015, 0.03))

        assert 'ERROR    palletjack.utils:utils.py:199 Did not receive a valid geocoding response; status code: 500' in caplog.text
        assert palletjack.utils.requests.get.call_count == 4
        assert result == (0, 0, 0., 'No API response')

    def test_geocode_addr_sleeps_appropriately(self, mocker):
        mocker.patch('palletjack.utils.requests', autospec=True)
        mocker.patch('palletjack.utils.sleep')

        response_mock = mocker.Mock()
        response_mock.json.return_value = {
            'status': 200,
            'result': {
                'location': {
                    'x': 123,
                    'y': 456
                },
                'score': 100.,
                'matchAddress': 'bar'
            }
        }
        response_mock.status_code = 200

        palletjack.utils.requests.get.return_value = response_mock

        palletjack.utils.geocode_addr('foo', 'bar', 'foo_key', (0.015, 0.03))

        palletjack.utils.sleep.assert_called_once()


class TestReportingIntervalModulus:

    def test_calc_modulus_for_reporting_interval_handles_less_than_ten(self):

        assert palletjack.utils.calc_modulus_for_reporting_interval(9) == 1

    def test_calc_modulus_for_reporting_interval_handles_ten(self):

        assert palletjack.utils.calc_modulus_for_reporting_interval(10) == 1

    def test_calc_modulus_for_reporting_interval_handles_less_than_split_value(self):

        assert palletjack.utils.calc_modulus_for_reporting_interval(100, split_value=500) == 10

    def test_calc_modulus_for_reporting_interval_handles_more_than_split_value(self):

        assert palletjack.utils.calc_modulus_for_reporting_interval(1000, split_value=500) == 50
