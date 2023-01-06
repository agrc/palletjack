import datetime
import logging
import re
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from arcgis import geometry
from pandas import testing as tm

import palletjack


@pytest.fixture(scope='module')  #: only call this once per module
def iris():
    return pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')


class TestRenameColumns:

    def test_rename_columns_for_agol_handles_special_and_space(self):
        cols = ['Test Name:']

        renamed = palletjack.utils.rename_columns_for_agol(cols)

        assert renamed == {'Test Name:': 'Test_Name_'}

    def test_rename_columns_for_agol_handles_special_and_space_leaves_others_alone(self):
        cols = ['Test Name:', 'FooName']

        renamed = palletjack.utils.rename_columns_for_agol(cols)

        assert renamed == {'Test Name:': 'Test_Name_', 'FooName': 'FooName'}

    def test_rename_columns_for_agol_retains_underscores(self):
        cols = ['Test Name:', 'Foo_Name']

        renamed = palletjack.utils.rename_columns_for_agol(cols)

        assert renamed == {'Test Name:': 'Test_Name_', 'Foo_Name': 'Foo_Name'}

    def test_rename_columns_for_agol_moves_leading_digits_to_end(self):
        cols = ['1TestName:', '12TestName:']

        renamed = palletjack.utils.rename_columns_for_agol(cols)

        assert renamed == {'1TestName:': 'TestName_1', '12TestName:': 'TestName_12'}

    def test_rename_columns_for_agol_moves_leading_underscore_to_end(self):
        cols = ['_TestName:']

        renamed = palletjack.utils.rename_columns_for_agol(cols)

        assert renamed == {'_TestName:': 'TestName__'}

    def test_rename_columns_for_agol_moves_underscore_after_leading_digits_to_end(self):
        cols = ['1_TestName:']

        renamed = palletjack.utils.rename_columns_for_agol(cols)

        assert renamed == {'1_TestName:': 'TestName_1_'}

    def test_rename_columns_for_agol_moves_underscore_before_leading_digits_to_end(self):
        cols = ['_1TestName:']

        renamed = palletjack.utils.rename_columns_for_agol(cols)

        assert renamed == {'_1TestName:': 'TestName__1'}


class TestCheckFieldsMatch:

    def test_check_fields_match_normal(self, mocker):
        mock_fl = mocker.Mock()
        mock_fl.properties.fields = [
            {
                'name': 'Foo'
            },
            {
                'name': 'Bar'
            },
        ]
        df = pd.DataFrame(columns=['Foo', 'Bar'])

        palletjack.utils.check_fields_match(mock_fl, df)

    def test_check_fields_match_raises_error_on_extra_new_field(self, mocker):
        mock_fl = mocker.Mock()
        mock_fl.properties.fields = [
            {
                'name': 'Foo'
            },
            {
                'name': 'Bar'
            },
        ]
        df = pd.DataFrame(columns=['Foo', 'Bar', 'Baz'])

        with pytest.raises(RuntimeError) as exc_info:
            palletjack.utils.check_fields_match(mock_fl, df)

        assert exc_info.value.args[
            0] == 'New dataset contains the following fields that are not present in the live dataset: {\'Baz\'}'

    def test_check_fields_match_ignores_new_shape_field(self, mocker):
        mock_fl = mocker.Mock()
        mock_fl.properties.fields = [
            {
                'name': 'Foo'
            },
            {
                'name': 'Bar'
            },
        ]
        df = pd.DataFrame(columns=['Foo', 'Bar', 'SHAPE'])

        palletjack.utils.check_fields_match(mock_fl, df)

    def test_check_fields_match_warns_on_missing_new_field(self, mocker, caplog):
        mock_fl = mocker.Mock()
        mock_fl.properties.fields = [
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

    def test_replace_nan_series_with_bogus_value_one_empty_one_non_empty_float(self, mocker):
        df = pd.DataFrame({
            'normal': [1., 2., 3.],
            'empty': [np.nan, np.nan, np.nan],
        })

        fixed_df = palletjack.utils.replace_nan_series_with_bogus_value(df)

        test_df = pd.DataFrame({
            'normal': [1., 2., 3.],
            'empty': [-999.9, -999.9, -999.9],
        })

        tm.assert_frame_equal(fixed_df, test_df)

    def test_replace_nan_series_with_bogus_value_other_series_has_nan(self, mocker):
        df = pd.DataFrame({
            'normal': [1., 2., np.nan],
            'empty': [np.nan, np.nan, np.nan],
        })

        fixed_df = palletjack.utils.replace_nan_series_with_bogus_value(df)

        test_df = pd.DataFrame({
            'normal': [1., 2., np.nan],
            'empty': [-999.9, -999.9, -999.9],
        })

        tm.assert_frame_equal(fixed_df, test_df)

    def test_replace_nan_series_with_bogus_value_other_series_is_empty_str(self, mocker):
        df = pd.DataFrame({
            'normal': ['', '', ''],
            'empty': [np.nan, np.nan, np.nan],
        })

        fixed_df = palletjack.utils.replace_nan_series_with_bogus_value(df)

        test_df = pd.DataFrame({
            'normal': ['', '', ''],
            'empty': [-999.9, -999.9, -999.9],
        })

        tm.assert_frame_equal(fixed_df, test_df)


class TestCheckIndexColumnInFL:

    def test_check_index_column_in_feature_layer_doesnt_raise_on_normal(self, mocker):
        fl_mock = mocker.Mock()
        fl_mock.properties.fields = [
            {
                'name': 'Foo'
            },
            {
                'name': 'Bar'
            },
        ]

        palletjack.utils.check_index_column_in_feature_layer(fl_mock, 'Foo')

    def test_check_index_column_in_feature_layer_raises_on_missing(self, mocker):
        fl_mock = mocker.Mock()
        fl_mock.properties.fields = [
            {
                'name': 'Foo'
            },
            {
                'name': 'Bar'
            },
        ]

        with pytest.raises(RuntimeError) as exc_info:
            palletjack.utils.check_index_column_in_feature_layer(fl_mock, 'Baz')

        assert exc_info.value.args[0] == 'Index column Baz not found in feature layer fields [\'Foo\', \'Bar\']'


class TestCheckFieldUnique:

    def test_check_field_set_to_unique_doesnt_raise_on_normal(self, mocker):
        mock_fl = mocker.Mock()
        mock_fl.properties.indexes = [
            {
                'fields': 'foo_field',
                'isUnique': True
            },
        ]

        palletjack.utils.check_field_set_to_unique(mock_fl, 'foo_field')

    def test_check_field_set_to_unique_raises_on_non_indexed_field(self, mocker):
        mock_fl = mocker.Mock()
        mock_fl.properties.indexes = [
            {
                'fields': 'bar_field',
                'isUnique': True
            },
        ]

        with pytest.raises(RuntimeError) as exc_info:
            palletjack.utils.check_field_set_to_unique(mock_fl, 'foo_field')

        assert exc_info.value.args[0] == 'foo_field does not have a "unique constraint" set within the feature layer'

    def test_check_field_set_to_unique_raises_on_non_unique_field(self, mocker):
        mock_fl = mocker.Mock()
        mock_fl.properties.indexes = [
            {
                'fields': 'foo_field',
                'isUnique': False
            },
        ]

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

        palletjack.utils.Geocoding.geocode_addr('foo', 'bar', 'foo_key', (0.015, 0.03))

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

        palletjack.utils.Geocoding.geocode_addr('foo', 'bar', 'foo_key', (0.015, 0.03), spatialReference=3857)

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

        result = palletjack.utils.Geocoding.geocode_addr('foo', 'bar', 'foo_key', (0.015, 0.03))

        assert result == (123, 456, 100., 'bar')

    def test_geocode_addr_returns_null_island_bad_status_code(self, mocker):
        mocker.patch('palletjack.utils.requests', autospec=True)
        mocker.patch('palletjack.utils.sleep')

        response_mock = mocker.Mock()
        response_mock.status_code = 404

        palletjack.utils.requests.get.return_value = response_mock

        result = palletjack.utils.Geocoding.geocode_addr('foo', 'bar', 'foo_key', (0.015, 0.03))

        assert result == (0, 0, 0., 'No Match')

    def test_geocode_addr_returns_null_island_on_404(self, mocker, caplog):
        mocker.patch('palletjack.utils.requests', autospec=True)
        mocker.patch('palletjack.utils.sleep')

        response_mock = mocker.Mock()
        response_mock.status_code = 404

        def bool_mock(self):
            if self.status_code < 400:
                return True
            return False

        response_mock.__bool__ = bool_mock

        palletjack.utils.requests.get.return_value = response_mock

        result = palletjack.utils.Geocoding.geocode_addr('foo', 'bar', 'foo_key', (0.015, 0.03))

        assert result == (0, 0, 0., 'No Match')

    def test_geocode_addr_404_doesnt_raise_no_response_error(self, mocker, caplog):
        mocker.patch('palletjack.utils.requests', autospec=True)
        mocker.patch('palletjack.utils.sleep')

        response_mock = mocker.Mock()
        response_mock.status_code = 404

        def bool_mock(self):
            if self.status_code < 400:
                return True
            return False

        response_mock.__bool__ = bool_mock

        palletjack.utils.requests.get.return_value = response_mock

        result = palletjack.utils.Geocoding.geocode_addr('foo', 'bar', 'foo_key', (0.015, 0.03))

        assert 'No response from GET; request timeout?' not in caplog.text

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

        result = palletjack.utils.Geocoding.geocode_addr('foo', 'bar', 'foo_key', (0.015, 0.03))

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

        result = palletjack.utils.Geocoding.geocode_addr('foo', 'bar', 'foo_key', (0.015, 0.03))

        assert 'No response from GET; request timeout?' in caplog.text
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

        result = palletjack.utils.Geocoding.geocode_addr('foo', 'bar', 'foo_key', (0.015, 0.03))

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

        result = palletjack.utils.Geocoding.geocode_addr('foo', 'bar', 'foo_key', (0.015, 0.03))

        assert 'Did not receive a valid geocoding response; status code: 500' in caplog.messages
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

        palletjack.utils.Geocoding.geocode_addr('foo', 'bar', 'foo_key', (0.015, 0.03))

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


class TestValidateAPIKey:

    def test_validate_api_key_good_key(self, mocker):
        req_mock = mocker.patch('palletjack.utils.requests', autospec=True)
        response_mock = mocker.Mock()
        response_mock.json.return_value = {'status': 200, 'message': 'foo'}
        req_mock.get.return_value = response_mock

        #: Should not raise
        palletjack.utils.Geocoding.validate_api_key('foo')

    def test_validate_api_key_bad_key(self, mocker):
        req_mock = mocker.patch('palletjack.utils.requests', autospec=True)
        response_mock = mocker.Mock()
        response_mock.json.return_value = {'status': 400, 'message': 'Invalid API key'}
        req_mock.get.return_value = response_mock

        with pytest.raises(ValueError, match=re.escape('API key validation failed: Invalid API key')):
            palletjack.utils.Geocoding.validate_api_key('foo')

    def test_validate_api_key_handles_network_exception(self, mocker, caplog):
        req_mock = mocker.patch('palletjack.utils.requests', autospec=True)
        mocker.patch('palletjack.utils.sleep')
        # caplog.set_level(logging.DEBUG)
        req_mock.get.side_effect = [IOError('Random Error')] * 4

        with pytest.raises(
            RuntimeError,
            match=re.escape('Could not determine key validity; check your API key and/or network connection')
        ) as exc_info:
            palletjack.utils.Geocoding.validate_api_key('foo')

        assert req_mock.get.call_count == 4
        assert 'Random Error' in str(exc_info.value.__cause__)

    def test_validate_api_key_handles_other_response(self, mocker):
        req_mock = mocker.patch('palletjack.utils.requests', autospec=True)
        response_mock = mocker.Mock()
        response_mock.json.return_value = {'status': 404, 'message': 'Weird Response'}
        req_mock.get.return_value = response_mock

        with pytest.warns(UserWarning, match=re.escape('Unhandled API key validation response 404: Weird Response')):
            palletjack.utils.Geocoding.validate_api_key('foo')


class TestFieldRenaming:

    def test_rename_fields_renames_all_fields(self):
        parcels_df = pd.DataFrame({
            'account_no': [1, 2, 3],
            'type': ['sf', 'mf', 'condo'],
        })

        field_mapping = {
            'account_no': 'PARCEL_ID',
            'type': 'class',
        }

        renamed_df = palletjack.utils.rename_fields(parcels_df, field_mapping)

        assert list(renamed_df.columns) == ['PARCEL_ID', 'class']

    def test_rename_fields_renames_some_fields(self):
        parcels_df = pd.DataFrame({
            'account_no': [1, 2, 3],
            'class': ['sf', 'mf', 'condo'],
        })

        field_mapping = {
            'account_no': 'PARCEL_ID',
        }

        renamed_df = palletjack.utils.rename_fields(parcels_df, field_mapping)

        assert list(renamed_df.columns) == ['PARCEL_ID', 'class']

    def test_rename_fields_raises_exception_for_missing_field(self):
        parcels_df = pd.DataFrame({
            'account_no': [1, 2, 3],
            'type': ['sf', 'mf', 'condo'],
        })

        field_mapping = {
            'account_no': 'PARCEL_ID',
            'TYPE': 'class',
        }

        with pytest.raises(ValueError) as exception_info:
            renamed_df = palletjack.utils.rename_fields(parcels_df, field_mapping)

            assert 'Field TYPE not found in dataframe.' in str(exception_info)


class TestAuthorization:

    def test_authorize_pygsheets_auths_from_file(self, mocker):
        pygsheets_mock = mocker.patch.object(palletjack.utils, 'pygsheets')
        pygsheets_mock.authorize.return_value = 'authed'

        client = palletjack.utils.authorize_pygsheets('file')

        assert pygsheets_mock.authorize.called_once_with('file')
        assert client == 'authed'

    def test_authorize_pygsheets_auths_from_custom_credentials(self, mocker, caplog):
        pygsheets_mock = mocker.patch.object(palletjack.utils, 'pygsheets')
        pygsheets_mock.authorize.side_effect = [FileNotFoundError, 'authed']

        caplog.set_level(logging.DEBUG, logger='palletjack.utils')
        caplog.clear()

        client = palletjack.utils.authorize_pygsheets('credentials')

        assert 'Credentials file not found, trying as environment variable' in [rec.message for rec in caplog.records]

        assert pygsheets_mock.authorize.call_count == 2
        assert pygsheets_mock.authorize.call_args_list == [
            mocker.call(service_file='credentials'),
            mocker.call(custom_credentials='credentials')
        ]
        assert client == 'authed'

    def test_authorize_pygsheets_raises_after_failing_both(self, mocker, caplog):
        pygsheets_mock = mocker.patch.object(palletjack.utils, 'pygsheets')
        pygsheets_mock.authorize.side_effect = [FileNotFoundError, IOError]

        caplog.set_level(logging.DEBUG, logger='palletjack.utils')
        caplog.clear()

        with pytest.raises(RuntimeError):
            client = palletjack.utils.authorize_pygsheets('credentials')

        assert 'Credentials file not found, trying as environment variable' in [rec.message for rec in caplog.records]

        assert pygsheets_mock.authorize.call_count == 2
        assert pygsheets_mock.authorize.call_args_list == [
            mocker.call(service_file='credentials'),
            mocker.call(custom_credentials='credentials')
        ]


class TestEmptyStringsAsNulls:

    def test_converts_empty_strings_to_null(self):
        FeatureSet = namedtuple('FeatureSet', ['features'])
        Feature = namedtuple('Feature', ['attributes'])
        feature_set = FeatureSet([Feature({
            'a': 'foo',
            'b': 'baz',
        }), Feature({
            'a': '',
            'b': '',
        })])
        fields = [{
            'name': 'a',
            'type': 'esriFieldTypeInteger',
            'nullable': False,
        }, {
            'name': 'b',
            'type': 'esriFieldTypeInteger',
            'nullable': True,
        }]

        fixed_feature_set = palletjack.utils.fix_numeric_empty_strings(feature_set, fields)
        fixed_feature = fixed_feature_set.features[1]

        assert fixed_feature.attributes['a'] == ''
        assert fixed_feature.attributes['b'] is None

    def test_fix_numeric_empty_strings_handles_both_missing_shape_info_fields(self):
        FeatureSet = namedtuple('FeatureSet', ['features'])
        Feature = namedtuple('Feature', ['attributes'])
        feature_set = FeatureSet([Feature({
            'a': 'foo',
            'b': 'baz',
        }), Feature({
            'a': '',
            'b': '',
        })])
        fields = [{
            'name': 'a',
            'type': 'esriFieldTypeInteger',
            'nullable': False,
        }, {
            'name': 'b',
            'type': 'esriFieldTypeInteger',
            'nullable': True,
        }, {
            'name': 'Shape__Length',
            'type': 'esriFieldTypeDouble',
            'nullable': True,
        }, {
            'name': 'Shape__Area',
            'type': 'esriFieldTypeDouble',
            'nullable': True,
        }]

        fixed_feature_set = palletjack.utils.fix_numeric_empty_strings(feature_set, fields)
        fixed_feature = fixed_feature_set.features[1]

        assert fixed_feature.attributes['a'] == ''
        assert fixed_feature.attributes['b'] is None

    def test_fix_numeric_empty_strings_handles_single_missing_shape_info_field(self):
        FeatureSet = namedtuple('FeatureSet', ['features'])
        Feature = namedtuple('Feature', ['attributes'])
        feature_set = FeatureSet([Feature({
            'a': 'foo',
            'b': 'baz',
        }), Feature({
            'a': '',
            'b': '',
        })])
        fields = [{
            'name': 'a',
            'type': 'esriFieldTypeInteger',
            'nullable': False,
        }, {
            'name': 'b',
            'type': 'esriFieldTypeInteger',
            'nullable': True,
        }, {
            'name': 'Shape__Length',
            'type': 'esriFieldTypeDouble',
            'nullable': True,
        }]

        fixed_feature_set = palletjack.utils.fix_numeric_empty_strings(feature_set, fields)
        fixed_feature = fixed_feature_set.features[1]

        assert fixed_feature.attributes['a'] == ''
        assert fixed_feature.attributes['b'] is None


class TestCheckFieldsMatch:

    def test_check_live_and_new_field_types_match_normal(self):
        new_df = pd.DataFrame({
            'ints': [1, 2, 3],
            'floats': [4., 5., 6.],
            'strings': ['a', 'b', 'c'],
            'OBJECTID': [11, 12, 13],
            'GlobalID': [
                'cc1cd617-1e55-4153-914d-8abb6ef22f24', '0f45d56f-249e-494a-863e-6b3999619bae',
                'd3a64873-8a09-4351-9ea0-802e450329ea'
            ],
            # 'SHAPE': [geometry.Geometry([0, 0])] * 3
        })

        properties = {
            'fields': [
                {
                    'name': 'OBJECTID',
                    'type': 'esriFieldTypeOID'
                },
                {
                    'name': 'strings',
                    'type': 'esriFieldTypeString'
                },
                {
                    'name': 'ints',
                    'type': 'esriFieldTypeInteger'
                },
                {
                    'name': 'floats',
                    'type': 'esriFieldTypeDouble'
                },
                {
                    'name': 'GlobalID',
                    'type': 'esriFieldTypeGlobalID'
                },
            ]
        }

        #: If it raises an error, it failed.
        checker = palletjack.utils.FieldChecker(properties, new_df)
        checker.check_live_and_new_field_types_match(['ints', 'floats', 'strings', 'OBJECTID', 'GlobalID'])

    def test_check_live_and_new_field_types_match_converted(self):
        new_df = pd.DataFrame({
            'ints': [1, 2, 3],
            'floats': [4.1, 5.1, 6.1],
            'strings': ['a', 'b', 'c'],
            'OBJECTID': [11, 12, 13],
            'GlobalID': [
                'cc1cd617-1e55-4153-914d-8abb6ef22f24', '0f45d56f-249e-494a-863e-6b3999619bae',
                'd3a64873-8a09-4351-9ea0-802e450329ea'
            ],
            # 'SHAPE': [geometry.Geometry([0, 0])] * 3
        }).convert_dtypes()

        properties = {
            'fields': [
                {
                    'name': 'OBJECTID',
                    'type': 'esriFieldTypeOID',
                },
                {
                    'name': 'strings',
                    'type': 'esriFieldTypeString',
                },
                {
                    'name': 'ints',
                    'type': 'esriFieldTypeInteger',
                },
                {
                    'name': 'floats',
                    'type': 'esriFieldTypeDouble',
                },
                {
                    'name': 'GlobalID',
                    'type': 'esriFieldTypeGlobalID',
                },
            ]
        }

        #: If it raises an error, it failed.
        checker = palletjack.utils.FieldChecker(properties, new_df)
        checker.check_live_and_new_field_types_match(['ints', 'floats', 'strings', 'OBJECTID', 'GlobalID'])

    def test_check_live_and_new_field_types_match_raises_on_incompatible(self):
        new_df = pd.DataFrame({
            'ints': [1, 2, 3],
        })

        properties = {'fields': [{'name': 'ints', 'type': 'esriFieldTypeDouble'}]}

        with pytest.raises(
            ValueError,
            match=re.escape(
                'Field type incompatibilities (field, live type, new type): [(\'ints\', \'esriFieldTypeDouble\', \'int64\')]'
            )
        ):
            checker = palletjack.utils.FieldChecker(properties, new_df)
            checker.check_live_and_new_field_types_match(['ints'])

    def test_check_live_and_new_field_types_match_raises_on_multiple_incompatible(self):
        new_df = pd.DataFrame({
            'ints': [1, 2, 3],
            'floats': [1.1, 1.2, 1.3],
        })

        properties = {
            'fields': [
                {
                    'name': 'ints',
                    'type': 'esriFieldTypeDouble'
                },
                {
                    'name': 'floats',
                    'type': 'esriFieldTypeInteger'
                },
            ]
        }

        with pytest.raises(
            ValueError,
            match=re.escape(
                'Field type incompatibilities (field, live type, new type): [(\'ints\', \'esriFieldTypeDouble\', \'int64\'), (\'floats\', \'esriFieldTypeInteger\', \'float64\')]'
            )
        ):
            checker = palletjack.utils.FieldChecker(properties, new_df)
            checker.check_live_and_new_field_types_match(['ints', 'floats'])

    def test_check_live_and_new_field_types_match_raises_on_notimplemented_esri_type(self):
        new_df = pd.DataFrame({
            'ints': [1, 2, 3],
        })

        properties = {'fields': [{'name': 'ints', 'type': 'esriFieldTypeDate'}]}

        with pytest.raises(NotImplementedError) as exc_info:
            checker = palletjack.utils.FieldChecker(properties, new_df)
            checker.check_live_and_new_field_types_match(['ints'])

        assert 'Live field "ints" type "esriFieldTypeDate" not yet mapped to a pandas dtype' in str(exc_info.value)

    def test_check_live_and_new_field_types_removes_SHAPE(self, mocker):
        geocheck_mock = mocker.patch('palletjack.utils.FieldChecker._check_geometry_types')
        new_df = pd.DataFrame({
            'ints': [1, 2, 3],
            'SHAPE': [geometry.Geometry([0, 0])] * 3,
        }).convert_dtypes()

        properties = {'fields': [{'name': 'ints', 'type': 'esriFieldTypeInteger'}]}

        #: If it raises an error, it failed.
        checker = palletjack.utils.FieldChecker(properties, new_df)
        checker.check_live_and_new_field_types_match(['ints', 'SHAPE'])
        geocheck_mock.assert_called_once()

    def test_check_geometry_types_normal(self):
        new_df = pd.DataFrame.spatial.from_xy(
            pd.DataFrame({
                'OBJECTID': [11, 12, 13],
                'x': [0, 0, 0],
                'y': [0, 0, 0],
            }), x_column='x', y_column='y'
        )

        properties = {'geometryType': 'esriGeometryPoint', 'fields': {'a': ['b']}}

        #: If it raises an error, it failed.
        checker = palletjack.utils.FieldChecker(properties, new_df)
        checker._check_geometry_types()

    def test_check_geometry_types_raises_on_multiple_types(self, mocker):
        new_df = mocker.Mock()
        new_df.columns = ['SHAPE']
        new_df.spatial.geometry_type = [1, 2]

        properties = {'geometryType': 'esriGeometryPoint', 'fields': {'a': ['b']}}

        with pytest.raises(ValueError) as exc_info:
            checker = palletjack.utils.FieldChecker(properties, new_df)
            checker._check_geometry_types()

        assert 'New dataframe has multiple geometry types' in str(exc_info.value)

    def test_check_geometry_types_raises_on_incompatible_type(self, mocker):
        new_df = mocker.Mock()
        new_df.columns = ['SHAPE']
        new_df.spatial.geometry_type = ['Polygon']

        properties = {'geometryType': 'esriGeometryPoint', 'fields': {'a': ['b']}}

        with pytest.raises(ValueError) as exc_info:
            checker = palletjack.utils.FieldChecker(properties, new_df)
            checker._check_geometry_types()

        assert 'New dataframe geometry type "Polygon" incompatible with live geometry type "esriGeometryPoint"' in str(
            exc_info.value
        )

    def test_check_geometry_types_raises_on_missing_SHAPE(self, mocker):
        new_df = mocker.Mock()
        new_df.columns = ['foo']

        checker_mock = mocker.Mock()
        checker_mock.new_dataframe = new_df

        with pytest.raises(ValueError) as exc_info:
            palletjack.utils.FieldChecker._check_geometry_types(checker_mock)

        assert 'New dataframe does not have a SHAPE column' in str(exc_info.value)


class TestFieldNullChecker:

    def test_check_for_non_null_fields_raises_on_null_data_in_nonnullable_field(self):
        properties = {
            'fields': [
                {
                    'name': 'regular',
                    'nullable': True,
                    'defaultValue': None,
                },
                {
                    'name': 'non-nullable',
                    'nullable': False,
                    'defaultValue': None,
                },
            ]
        }
        new_df = pd.DataFrame({
            'regular': ['a', 'b'],
            'non-nullable': ['c', None],
        })

        checker = palletjack.utils.FieldChecker(properties, new_df)

        with pytest.raises(ValueError) as exc_info:
            checker.check_for_non_null_fields(['regular', 'non-nullable'])

        assert 'The following fields cannot have null values in the live data but one or more nulls exist in the new data: non-nullable' in str(
            exc_info.value
        )

    def test_check_for_non_null_fields_doesnt_raise_on_null_in_nullable_field(self):
        properties = {
            'fields': [
                {
                    'name': 'regular',
                    'nullable': True,
                    'defaultValue': None,
                },
            ]
        }
        new_df = pd.DataFrame({
            'regular': ['a', None],
        })

        checker = palletjack.utils.FieldChecker(properties, new_df)

        #: Should not raise an error
        checker.check_for_non_null_fields(['regular'])

    def test_check_for_non_null_fields_doesnt_raise_on_null_in_nonnullable_with_default(self):
        properties = {
            'fields': [
                {
                    'name': 'regular',
                    'nullable': True,
                    'defaultValue': 'foo',
                },
            ]
        }
        new_df = pd.DataFrame({
            'regular': ['a', None],
        })

        checker = palletjack.utils.FieldChecker(properties, new_df)

        #: Should not raise an error
        checker.check_for_non_null_fields(['regular'])

    def test_check_for_non_null_fields_skips_field(self):
        properties = {
            'fields': [
                {
                    'name': 'regular',
                    'nullable': True,
                    'defaultValue': None,
                },
                {
                    'name': 'non-nullable',
                    'nullable': False,
                    'defaultValue': None,
                },
            ]
        }
        new_df = pd.DataFrame({
            'regular': ['a', 'b'],
            'non-nullable': ['c', None],
        })

        checker = palletjack.utils.FieldChecker(properties, new_df)

        #: Should not raise
        checker.check_for_non_null_fields(['regular'])


class TestFieldLength:

    def test_check_field_length_normal_string(self):
        properties = {
            'fields': [
                {
                    'name': 'foo',
                    'type': 'esriFieldTypeString',
                    'length': 10,
                },
            ]
        }
        new_df = pd.DataFrame({
            'foo': ['aaa', 'bbbb'],
        })

        checker = palletjack.utils.FieldChecker(properties, new_df)

        #: Should not raise
        checker.check_field_length(['foo'])

    def test_check_field_length_raises_on_long_string(self):
        properties = {
            'fields': [
                {
                    'name': 'foo',
                    'type': 'esriFieldTypeString',
                    'length': 10,
                },
            ]
        }
        new_df = pd.DataFrame({
            'foo': ['aaa', 'bbbb', 'this string is far too long'],
        })

        checker = palletjack.utils.FieldChecker(properties, new_df)

        with pytest.raises(ValueError) as exc_info:
            checker.check_field_length(['foo'])

        assert 'Row 2, column foo in new data exceeds the live data max length of 10' in str(exc_info.value)

    def test_check_field_length_uses_fields_arg(self):
        properties = {
            'fields': [
                {
                    'name': 'foo',
                    'type': 'esriFieldTypeString',
                    'length': 10,
                },
                {
                    'name': 'bar',
                    'type': 'esriFieldTypeString',
                    'length': 10,
                },
            ]
        }
        new_df = pd.DataFrame({
            'foo': ['aaa', 'bbbb'],
            'bar': ['a', 'way too long field'],
        })

        checker = palletjack.utils.FieldChecker(properties, new_df)

        #: bar shouldn't trigger an exception
        checker.check_field_length(['foo'])

    def test_check_field_length_uses_ignores_new_field_not_in_live_data(self):
        properties = {
            'fields': [
                {
                    'name': 'foo',
                    'type': 'esriFieldTypeString',
                    'length': 10,
                },
            ]
        }
        new_df = pd.DataFrame({
            'foo': ['aaa', 'bbbb'],
            'bar': ['a', 'way too long field'],
        })

        checker = palletjack.utils.FieldChecker(properties, new_df)

        #: bar shouldn't trigger an exception
        checker.check_field_length(['foo'])

    def test_check_field_length_works_with_int_field(self):
        properties = {
            'fields': [
                {
                    'name': 'foo',
                    'type': 'esriFieldTypeString',
                    'length': 10,
                },
                {
                    'name': 'int',
                    'type': 'esriFieldTypeInteger',
                },
            ]
        }
        new_df = pd.DataFrame({
            'foo': ['aaa', 'bbbb'],
            'int': [1, 2],
        })

        checker = palletjack.utils.FieldChecker(properties, new_df)

        #: Should not raise
        checker.check_field_length(['foo', 'int'])

    def test_check_field_length_passes_with_only_int_fields(self, caplog):
        caplog.set_level(logging.DEBUG)
        properties = {
            'fields': [
                {
                    'name': 'foo',
                    'type': 'esriFieldTypeInteger',
                },
                {
                    'name': 'int',
                    'type': 'esriFieldTypeInteger',
                },
            ]
        }
        new_df = pd.DataFrame({
            'foo': ['aaa', 'bbbb'],
            'int': [1, 2],
        })

        checker = palletjack.utils.FieldChecker(properties, new_df)

        #: Should not raise
        checker.check_field_length(['foo'])

        assert 'No fields with length property' in caplog.text


class TestFieldsPresent:

    def test_check_fields_present_normal_in_both(self):
        properties = {'fields': [{'name': 'foo'}, {'name': 'bar'}]}

        new_df = pd.DataFrame(columns=['foo', 'bar'])

        checker = palletjack.utils.FieldChecker(properties, new_df)

        #: Should not raise
        checker.check_fields_present(['foo', 'bar'], False)

    def test_check_fields_present_raises_missing_live(self):
        properties = {'fields': [{'name': 'foo'}]}

        new_df = pd.DataFrame(columns=['foo', 'bar'])

        checker = palletjack.utils.FieldChecker(properties, new_df)

        with pytest.raises(RuntimeError) as exc_info:
            checker.check_fields_present(['foo', 'bar'], False)
        assert 'Fields missing in live data: bar' in str(exc_info.value)

    def test_check_fields_present_raises_missing_new(self):
        properties = {'fields': [{'name': 'foo'}, {'name': 'bar'}]}

        new_df = pd.DataFrame(columns=['foo'])

        checker = palletjack.utils.FieldChecker(properties, new_df)

        with pytest.raises(RuntimeError) as exc_info:
            checker.check_fields_present(['foo', 'bar'], False)
        assert 'Fields missing in new data: bar' in str(exc_info.value)

    def test_check_fields_present_raises_missing_both(self):
        properties = {'fields': [{'name': 'foo'}]}

        new_df = pd.DataFrame(columns=['bar'])

        checker = palletjack.utils.FieldChecker(properties, new_df)

        with pytest.raises(RuntimeError) as exc_info:
            checker.check_fields_present(['foo', 'bar'], False)
        assert 'Fields missing in live data: bar. Fields missing in new data: foo' in str(exc_info.value)

    def test_check_fields_present_adds_oid_to_list_of_fields_to_check(self):
        properties = {'fields': [{'name': 'foo'}, {'name': 'bar'}, {'name': 'OBJECTID'}]}

        new_df = pd.DataFrame(columns=['foo', 'bar', 'OBJECTID'])

        checker = palletjack.utils.FieldChecker(properties, new_df)

        #: Should not raise
        checker.check_fields_present(['foo', 'bar'], True)


class TestEmptyFieldWarnings:

    def test_check_for_empty_float_fields_doesnt_raise_on_no_empty_fields(self, mocker):
        properties = {'fields': ['int, float']}
        new_df = pd.DataFrame({'int': [1, 2, 3], 'float': [1.1, 1.2, 1.3]})

        checker = palletjack.utils.FieldChecker(properties, new_df)

        #: Should not raise
        checker.check_for_empty_float_fields(['int', 'float'])

    def test_check_for_empty_float_fields_raises_on_single_field(self, mocker):
        properties = {'fields': ['int, float']}
        new_df = pd.DataFrame({'int': [1, 2, 3], 'float': [np.nan] * 3})

        checker = palletjack.utils.FieldChecker(properties, new_df)

        with pytest.raises(
            ValueError,
            match=re.escape(
                'The following float/double column(s) are completely empty: [\'float\'] (suggestion: insert at least one bogus value)'
            )
        ):
            checker.check_for_empty_float_fields(['int', 'float'])

    def test_check_for_empty_float_fields_raises_on_multiple_fields(self, mocker):
        properties = {'fields': ['foo, float']}
        new_df = pd.DataFrame({'foo': [np.nan] * 3, 'float': [np.nan] * 3})

        checker = palletjack.utils.FieldChecker(properties, new_df)

        with pytest.raises(
            ValueError,
            match=re.escape(
                'The following float/double column(s) are completely empty: [\'foo\', \'float\'] (suggestion: insert at least one bogus value)'
            )
        ):
            checker.check_for_empty_float_fields(['foo', 'float'])


class TestNullGeometryGenerators:

    def test_get_null_geometries_point(self):
        properties = {'geometryType': 'esriGeometryPoint'}

        nullo = palletjack.utils.get_null_geometries(properties)

        assert nullo == '{"x": 0, "y": 0, "spatialReference": {"wkid": 4326}}'

    def test_get_null_geometries_polyline(self):
        properties = {'geometryType': 'esriGeometryPolyline'}

        nullo = palletjack.utils.get_null_geometries(properties)

        assert nullo == '{"paths": [[[0, 0], [0.1, 0.1], [0.2, 0.2]]], "spatialReference": {"wkid": 4326}}'

    def test_get_null_geometries_polygon(self):
        properties = {'geometryType': 'esriGeometryPolygon'}

        nullo = palletjack.utils.get_null_geometries(properties)

        assert nullo == '{"rings": [[[0, 0.1], [0.1, 0.1], [0.1, 0], [0, 0]]], "spatialReference": {"wkid": 4326}}'

    def test_get_null_geometries_raises_on_other(self):
        properties = {'geometryType': 'other'}

        with pytest.raises(NotImplementedError) as exc_info:
            nullo = palletjack.utils.get_null_geometries(properties)

        assert 'Null value generator for live geometry type other not yet implemented' in str(exc_info.value)


class TestDeleteUtils:

    def test_check_delete_oids_are_ints_casts_string_list_with_spaces(self):
        oid_list = [' 1', ' 2', '3']

        numeric_oids = palletjack.utils.DeleteUtils.check_delete_oids_are_ints(oid_list)

        assert numeric_oids == [1, 2, 3]

    def test_check_delete_oids_are_ints_casts_string_list_without_spaces(self):
        oid_list = ['1', '2', '3']

        numeric_oids = palletjack.utils.DeleteUtils.check_delete_oids_are_ints(oid_list)

        assert numeric_oids == [1, 2, 3]

    def test_check_delete_oids_are_ints_passes_on_int_list(self):
        oid_list = [1, 2, 3]

        numeric_oids = palletjack.utils.DeleteUtils.check_delete_oids_are_ints(oid_list)

        assert numeric_oids == [1, 2, 3]

    def test_check_delete_oids_are_ints_raises_first_non_int(self):
        oid_list = ['one', 2, 3]

        with pytest.raises(TypeError) as exc_info:
            palletjack.utils.DeleteUtils.check_delete_oids_are_ints(oid_list)

        assert 'Couldn\'t convert OBJECTID(s) `[\'one\']` to integer' in str(exc_info.value)

    def test_check_delete_oids_are_ints_raises_later_non_ints(self):
        oid_list = [1, 'two', 'three']

        with pytest.raises(TypeError) as exc_info:
            palletjack.utils.DeleteUtils.check_delete_oids_are_ints(oid_list)

        assert 'Couldn\'t convert OBJECTID(s) `[\'two\', \'three\']` to integer' in str(exc_info.value)

    def test_check_for_empty_oid_list_doesnt_raise_on_list(self):
        oid_list = [1, 2, 3]
        numeric_oids = [1, 2, 3]

        palletjack.utils.DeleteUtils.check_for_empty_oid_list(oid_list, numeric_oids)

    def test_check_for_empty_oid_list_raises_on_empty_list(self):
        oid_list = []
        numeric_oids = []

        with pytest.raises(ValueError) as exc_info:
            palletjack.utils.DeleteUtils.check_for_empty_oid_list(oid_list, numeric_oids)

        assert 'No OBJECTIDs found in []' in str(exc_info.value)

    def test_check_delete_oids_are_in_live_data_doesnt_warn_on_good_oids(self, mocker):
        fl_mock = mocker.Mock()
        fl_mock.query.return_value = {'objectIdFieldName': 'OBJECTID', 'objectIds': [1, 2, 3]}
        oid_list = [1, 2, 3]
        oid_numeric = [1, 2, 3]

        palletjack.utils.DeleteUtils.check_delete_oids_are_in_live_data(oid_list, oid_numeric, fl_mock)

    def test_check_delete_oids_are_in_live_data_warns_on_missing_oid(self, mocker):
        fl_mock = mocker.Mock()
        fl_mock.query.return_value = {'objectIdFieldName': 'OBJECTID', 'objectIds': [1, 2]}
        oid_list = [1, 2, 3]
        oid_numeric = [1, 2, 3]

        with pytest.warns(UserWarning, match=re.escape('OBJECTIDs {3} were not found in the live data')):
            number_of_missing_oids = palletjack.utils.DeleteUtils.check_delete_oids_are_in_live_data(
                oid_list, oid_numeric, fl_mock
            )

        assert number_of_missing_oids == 1


class TestSaveDataframeToJSON:

    def test_save_feature_layer_to_json_calls_write_with_json_and_returns_right_path(self, mocker):

        mock_df = pd.DataFrame({
            'foo': [1],
            'x': [11],
            'y': [14],
        })

        mock_sdf = pd.DataFrame.spatial.from_xy(mock_df, 'x', 'y')
        mock_fl = mocker.Mock()
        mock_fl.query.return_value.sdf = mock_sdf
        mock_fl.properties.name = 'flayer'

        open_mock = mocker.MagicMock()
        context_manager_mock = mocker.MagicMock()
        context_manager_mock.return_value.__enter__.return_value = open_mock
        mocker.patch('palletjack.utils.Path.open', new=context_manager_mock)

        out_path = palletjack.utils.save_feature_layer_to_json(mock_fl, 'foo')

        test_json_string = '{"features": [{"geometry": {"spatialReference": {"wkid": 4326}, "x": 11, "y": 14}, "attributes": {"foo": 1, "x": 11, "y": 14, "OBJECTID": 1}}], "objectIdFieldName": "OBJECTID", "displayFieldName": "OBJECTID", "spatialReference": {"wkid": 4326}, "geometryType": "esriGeometryPoint", "fields": [{"name": "OBJECTID", "type": "esriFieldTypeOID", "alias": "OBJECTID"}, {"name": "foo", "type": "esriFieldTypeDouble", "alias": "foo"}, {"name": "x", "type": "esriFieldTypeDouble", "alias": "x"}, {"name": "y", "type": "esriFieldTypeDouble", "alias": "y"}]}'

        open_mock.write.assert_called_with(test_json_string)
        assert out_path == Path('foo', f'flayer_{datetime.date.today()}.json')


class TestDataFrameChunking:

    def test_ceildiv_always_returns_at_least_one(self):
        assert palletjack.utils._ceildiv(1, 2) == 1

    def test_ceildiv_returns_ceiling(self):
        assert palletjack.utils._ceildiv(3, 2) == 2

    def test_chunk_dataframe_properly_chunks(self, iris):

        chunks = 4
        chunk_size = palletjack.utils._ceildiv(len(iris), chunks)
        dfs = palletjack.utils._chunk_dataframe(iris, chunk_size)

        assert len(dfs) == chunks
        assert [len(df) for df in dfs] == [38, 38, 38, 36]

    def test_chunk_dataframe_properly_chunks_even_sized_chunks(self, iris):

        chunks = 5
        chunk_size = palletjack.utils._ceildiv(len(iris), chunks)
        dfs = palletjack.utils._chunk_dataframe(iris, chunk_size)

        assert len(dfs) == chunks
        assert [len(df) for df in dfs] == [30] * chunks

    def test_recursive_dataframe_chunking_recurses_on_first_chunk_too_large(self, mocker):
        #: 5 rows, 2 chunks, first chunk gets 4 instead of three.
        #: two total calls: once for 5 elements, once for 4. first call chunks to 4 and 1, next chunks 4 into 3 and 1
        df = pd.DataFrame(['a', 'b', 'c', 'd', 'e'], columns=['foo'])
        mocker.patch('palletjack.utils.pd.DataFrame.spatial.to_featureset', return_value=mocker.Mock())
        mocker.patch('palletjack.utils._ceildiv')
        sys_mock = mocker.patch('palletjack.utils.sys.getsizeof', side_effect=['foo', 4, 'foo', 2, 2, 1])
        mocker.patch(
            'palletjack.utils._chunk_dataframe',
            side_effect=[
                [df.iloc[:4], df.iloc[4:]],  #: first chunking gives 4 and 1
                [df.iloc[:3], df.iloc[3:4]],  #: second chunking gives 3 and 1 (3 being the max size)
            ]
        )
        df_list = palletjack.utils._recursive_dataframe_chunking(df, 3)

        tm.assert_frame_equal(df_list[0], df.iloc[:3])
        tm.assert_frame_equal(df_list[1], df.iloc[3:4])
        tm.assert_frame_equal(df_list[2], df.iloc[4:])
        assert sys_mock.call_count == 6

    def test_recursive_dataframe_chunking_recurses_on_second_chunk_too_large(self, mocker):
        #: 5 rows, 2 chunks, first chunk gets 1 instead of three, second gets 4
        #: two total calls: once for 5 elements, once for 4. first call chunks to 1 and 4, next chunks 4 into 3 and 1
        df = pd.DataFrame(['a', 'b', 'c', 'd', 'e'], columns=['foo'])
        mocker.patch('palletjack.utils.pd.DataFrame.spatial.to_featureset', return_value=mocker.Mock())
        mocker.patch('palletjack.utils._ceildiv')
        sys_mock = mocker.patch('palletjack.utils.sys.getsizeof', side_effect=['foo', 1, 4, 'foo', 2, 2])
        mocker.patch(
            'palletjack.utils._chunk_dataframe',
            side_effect=[
                [df.iloc[:1], df.iloc[1:]],  #: first chunking gives 1 and 4
                [df.iloc[1:4], df.iloc[4:]],  #: second chunking gives 3 and 1 (3 being the max size)
            ]
        )

        df_list = palletjack.utils._recursive_dataframe_chunking(df, 3)

        tm.assert_frame_equal(df_list[0], df.iloc[:1])
        tm.assert_frame_equal(df_list[1], df.iloc[1:4])
        tm.assert_frame_equal(df_list[2], df.iloc[4:])
        assert sys_mock.call_count == 6

    def test_recursive_dataframe_chunking_recurses_on_middle_chunk_too_large(self, mocker):
        #: 5 rows, 3 chunks, first chunk gets 1, second gets 3, third gets 1
        #: two total calls: once for 5 elements, once for 3.
        df = pd.DataFrame(['a', 'b', 'c', 'd', 'e'], columns=['foo'])
        mocker.patch('palletjack.utils.pd.DataFrame.spatial.to_featureset', return_value=mocker.Mock())
        mocker.patch('palletjack.utils._ceildiv')
        sys_mock = mocker.patch('palletjack.utils.sys.getsizeof', side_effect=['foo', 1, 3, 'foo', 2, 1, 1])
        mocker.patch(
            'palletjack.utils._chunk_dataframe',
            side_effect=[
                [df.iloc[:1], df.iloc[1:4], df.iloc[4:]],  #: first chunking gives 1, 3, 1
                [df.iloc[1:3], df.iloc[3:4]],  #: second chunking gives 2 and 1 (2 being the max size)
            ]
        )

        df_list = palletjack.utils._recursive_dataframe_chunking(df, 2)

        tm.assert_frame_equal(df_list[0], df.iloc[:1])
        tm.assert_frame_equal(df_list[1], df.iloc[1:3])
        tm.assert_frame_equal(df_list[2], df.iloc[3:4])
        tm.assert_frame_equal(df_list[3], df.iloc[4:])
        assert sys_mock.call_count == 7

    def test_recursive_dataframe_chunking_doesnt_recurse_when_not_needed(self, mocker):
        #: 5 rows, 2 chunks, first chunk gets 3, second gets 2
        #: only one call: first breaks into 3 and 2
        df = pd.DataFrame(['a', 'b', 'c', 'd', 'e'], columns=['foo'])
        mocker.patch('palletjack.utils.pd.DataFrame.spatial.to_featureset', return_value=mocker.Mock())
        mocker.patch('palletjack.utils._ceildiv')
        sys_mock = mocker.patch('palletjack.utils.sys.getsizeof', side_effect=['foo', 3, 2])
        mocker.patch(
            'palletjack.utils._chunk_dataframe',
            side_effect=[
                [df.iloc[:3], df.iloc[3:]],  #: first chunking gives 3, 2
            ]
        )

        df_list = palletjack.utils._recursive_dataframe_chunking(df, 3)

        tm.assert_frame_equal(df_list[0], df.iloc[:3])
        tm.assert_frame_equal(df_list[1], df.iloc[3:])
        assert sys_mock.call_count == 3

    def test_recursive_dataframe_chunking_doesnt_chunk_when_not_needed(self, mocker):
        #: 5 rows, 2 chunks, first chunk gets 3, second gets 2
        #: only one call: first breaks into 3 and 2
        df = pd.DataFrame(['a', 'b', 'c', 'd', 'e'], columns=['foo'])
        mocker.patch('palletjack.utils.pd.DataFrame.spatial.to_featureset', return_value=mocker.Mock())
        mocker.patch('palletjack.utils._ceildiv')
        sys_mock = mocker.patch('palletjack.utils.sys.getsizeof', side_effect=['foo', 5])
        chunking_mock = mocker.patch(
            'palletjack.utils._chunk_dataframe',
            side_effect=[
                [df],  #: no chunk needed
            ]
        )

        df_list = palletjack.utils._recursive_dataframe_chunking(df, 5)

        tm.assert_frame_equal(df_list[0], df)
        chunking_mock.assert_called_once()
        assert sys_mock.call_count == 2

    def test_build_upload_json_calls_null_string_fixer_appropriate_number_of_times(self, mocker):

        mock_df = mocker.Mock()
        mock_string_fixer = mocker.patch('palletjack.utils.fix_numeric_empty_strings')
        mock_string_fixer.return_value.to_geojson = 'new_json'
        mocker.patch('palletjack.utils.sys.getsizeof')
        mocker.patch('palletjack.utils._recursive_dataframe_chunking', return_value=[mocker.Mock()] * 5)

        json_list = palletjack.utils.build_upload_json(mock_df, 'foo_dict')

        assert len(json_list) == 5
        assert json_list == ['new_json'] * 5
        assert mock_string_fixer.call_count == 5
