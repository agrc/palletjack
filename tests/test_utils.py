import datetime
import logging
import re
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
import pytest
from arcgis import geometry
from arcgis.features import FeatureLayer, Table
from pandas import testing as tm

import palletjack


@pytest.fixture(scope="module")  #: only call this once per module
def iris():
    return pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")


@pytest.fixture
def set_max_tries():
    palletjack.utils.RETRY_MAX_TRIES = 1
    yield
    palletjack.utils.RETRY_MAX_TRIES = 3


@pytest.fixture
def set_delay_time():
    palletjack.utils.RETRY_DELAY_TIME = 3
    yield
    palletjack.utils.RETRY_DELAY_TIME = 2


class TestRenameColumns:
    def test_rename_columns_for_agol_handles_special_and_space(self):
        cols = ["Test Name:"]

        renamed = palletjack.utils.rename_columns_for_agol(cols)

        assert renamed == {"Test Name:": "Test_Name_"}

    def test_rename_columns_for_agol_handles_special_and_space_leaves_others_alone(self):
        cols = ["Test Name:", "FooName"]

        renamed = palletjack.utils.rename_columns_for_agol(cols)

        assert renamed == {"Test Name:": "Test_Name_", "FooName": "FooName"}

    def test_rename_columns_for_agol_retains_underscores(self):
        cols = ["Test Name:", "Foo_Name"]

        renamed = palletjack.utils.rename_columns_for_agol(cols)

        assert renamed == {"Test Name:": "Test_Name_", "Foo_Name": "Foo_Name"}

    def test_rename_columns_for_agol_moves_leading_digits_to_end(self):
        cols = ["1TestName:", "12TestName:"]

        renamed = palletjack.utils.rename_columns_for_agol(cols)

        assert renamed == {"1TestName:": "TestName_1", "12TestName:": "TestName_12"}

    def test_rename_columns_for_agol_moves_leading_underscore_to_end(self):
        cols = ["_TestName:"]

        renamed = palletjack.utils.rename_columns_for_agol(cols)

        assert renamed == {"_TestName:": "TestName__"}

    def test_rename_columns_for_agol_moves_underscore_after_leading_digits_to_end(self):
        cols = ["1_TestName:"]

        renamed = palletjack.utils.rename_columns_for_agol(cols)

        assert renamed == {"1_TestName:": "TestName_1_"}

    def test_rename_columns_for_agol_moves_underscore_before_leading_digits_to_end(self):
        cols = ["_1TestName:"]

        renamed = palletjack.utils.rename_columns_for_agol(cols)

        assert renamed == {"_1TestName:": "TestName__1"}


class TestRetry:
    def test_retry_returns_on_first_success(self, mocker):
        mock = mocker.Mock()
        mock.function.return_value = 42

        answer = palletjack.utils.retry(mock.function, "a", "b")

        assert answer == 42
        assert mock.function.call_count == 1

    def test_retry_returns_after_one_failure(self, mocker):
        mock = mocker.Mock()
        mock.function.side_effect = [Exception, 42]
        mocker.patch("palletjack.utils.sleep")

        answer = palletjack.utils.retry(mock.function, "a", "b")

        assert answer == 42
        assert mock.function.call_count == 2

    def test_retry_returns_after_two_failures(self, mocker):
        mock = mocker.Mock()
        mock.function.side_effect = [
            Exception,
            Exception,
            42,
        ]
        mocker.patch("palletjack.utils.sleep")

        answer = palletjack.utils.retry(mock.function, "a", "b")

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
        mocker.patch("palletjack.utils.sleep")

        with pytest.raises(Exception):
            answer = palletjack.utils.retry(mock.function, "a", "b")

        assert mock.function.call_count == 4

    def test_retry_uses_global_retry_max_value(self, mocker, set_max_tries):
        mock = mocker.Mock()
        mock.function.side_effect = [Exception, Exception, 42]
        mocker.patch("palletjack.utils.sleep")

        with pytest.raises(Exception):
            answer = palletjack.utils.retry(mock.function, "a", "b")

        assert mock.function.call_count == 2

    def test_retry_uses_global_retry_delay_time_value(self, mocker, set_delay_time):
        mock = mocker.Mock()
        mock.function.side_effect = [
            Exception,
            42,
        ]
        sleep_mock = mocker.patch("palletjack.utils.sleep")

        answer = palletjack.utils.retry(mock.function, "a", "b")

        assert answer == 42
        sleep_mock.assert_called_once_with(3)


class TestCheckIndexColumnInFL:
    def test_check_index_column_in_feature_layer_doesnt_raise_on_normal(self, mocker):
        fl_mock = mocker.Mock()
        fl_mock.properties.fields = [
            {"name": "Foo"},
            {"name": "Bar"},
        ]

        palletjack.utils.check_index_column_in_feature_layer(fl_mock, "Foo")

    def test_check_index_column_in_feature_layer_raises_on_missing(self, mocker):
        fl_mock = mocker.Mock()
        fl_mock.properties.fields = [
            {"name": "Foo"},
            {"name": "Bar"},
        ]

        with pytest.raises(RuntimeError) as exc_info:
            palletjack.utils.check_index_column_in_feature_layer(fl_mock, "Baz")

        assert exc_info.value.args[0] == "Index column Baz not found in feature layer fields ['Foo', 'Bar']"


class TestCheckFieldUnique:
    def test_check_field_set_to_unique_doesnt_raise_on_normal(self, mocker):
        mock_fl = mocker.Mock()
        mock_fl.properties.indexes = [
            {"fields": "foo_field", "isUnique": True},
        ]

        palletjack.utils.check_field_set_to_unique(mock_fl, "foo_field")

    def test_check_field_set_to_unique_raises_on_non_indexed_field(self, mocker):
        mock_fl = mocker.Mock()
        mock_fl.properties.indexes = [
            {"fields": "bar_field", "isUnique": True},
        ]

        with pytest.raises(RuntimeError) as exc_info:
            palletjack.utils.check_field_set_to_unique(mock_fl, "foo_field")

        assert exc_info.value.args[0] == 'foo_field does not have a "unique constraint" set within the feature layer'

    def test_check_field_set_to_unique_raises_on_non_unique_field(self, mocker):
        mock_fl = mocker.Mock()
        mock_fl.properties.indexes = [
            {"fields": "foo_field", "isUnique": False},
        ]

        with pytest.raises(RuntimeError) as exc_info:
            palletjack.utils.check_field_set_to_unique(mock_fl, "foo_field")

        assert exc_info.value.args[0] == 'foo_field does not have a "unique constraint" set within the feature layer'


class TestGeocodeAddr:
    def test_geocode_addr_builds_url_correctly(self, mocker):
        mocker.patch("palletjack.utils.requests", autospec=True)
        mocker.patch("palletjack.utils.sleep")

        response_mock = mocker.Mock()
        response_mock.json.return_value = {
            "status": 200,
            "result": {"location": {"x": 123, "y": 456}, "score": 100.0, "matchAddress": "bar"},
        }
        response_mock.status_code = 200

        palletjack.utils.requests.get.return_value = response_mock

        palletjack.utils.Geocoding.geocode_addr("foo", "bar", "foo_key", (0.015, 0.03))

        palletjack.utils.requests.get.assert_called_with(
            "https://api.mapserv.utah.gov/api/v1/geocode/foo/bar", params={"apiKey": "foo_key"}
        )

    def test_geocode_addr_handles_kwargs_for_geocoding_api(self, mocker):
        mocker.patch("palletjack.utils.requests", autospec=True)
        mocker.patch("palletjack.utils.sleep")

        response_mock = mocker.Mock()
        response_mock.json.return_value = {
            "status": 200,
            "result": {"location": {"x": 123, "y": 456}, "score": 100.0, "matchAddress": "bar"},
        }
        response_mock.status_code = 200

        palletjack.utils.requests.get.return_value = response_mock

        palletjack.utils.Geocoding.geocode_addr("foo", "bar", "foo_key", (0.015, 0.03), spatialReference=3857)

        palletjack.utils.requests.get.assert_called_with(
            "https://api.mapserv.utah.gov/api/v1/geocode/foo/bar",
            params={"apiKey": "foo_key", "spatialReference": 3857},
        )

    def test_geocode_addr_returns_coords(self, mocker):
        mocker.patch("palletjack.utils.requests", autospec=True)
        mocker.patch("palletjack.utils.sleep")

        response_mock = mocker.Mock()
        response_mock.json.return_value = {
            "status": 200,
            "result": {"location": {"x": 123, "y": 456}, "score": 100.0, "matchAddress": "bar"},
        }
        response_mock.status_code = 200

        palletjack.utils.requests.get.return_value = response_mock

        result = palletjack.utils.Geocoding.geocode_addr("foo", "bar", "foo_key", (0.015, 0.03))

        assert result == (123, 456, 100.0, "bar")

    def test_geocode_addr_returns_null_island_bad_status_code(self, mocker):
        mocker.patch("palletjack.utils.requests", autospec=True)
        mocker.patch("palletjack.utils.sleep")

        response_mock = mocker.Mock()
        response_mock.status_code = 404

        palletjack.utils.requests.get.return_value = response_mock

        result = palletjack.utils.Geocoding.geocode_addr("foo", "bar", "foo_key", (0.015, 0.03))

        assert result == (0, 0, 0.0, "No Match")

    def test_geocode_addr_returns_null_island_on_404(self, mocker, caplog):
        mocker.patch("palletjack.utils.requests", autospec=True)
        mocker.patch("palletjack.utils.sleep")

        response_mock = mocker.Mock()
        response_mock.status_code = 404

        def bool_mock(self):
            if self.status_code < 400:
                return True
            return False

        response_mock.__bool__ = bool_mock

        palletjack.utils.requests.get.return_value = response_mock

        result = palletjack.utils.Geocoding.geocode_addr("foo", "bar", "foo_key", (0.015, 0.03))

        assert result == (0, 0, 0.0, "No Match")

    def test_geocode_addr_404_doesnt_raise_no_response_error(self, mocker, caplog):
        mocker.patch("palletjack.utils.requests", autospec=True)
        mocker.patch("palletjack.utils.sleep")

        response_mock = mocker.Mock()
        response_mock.status_code = 404

        def bool_mock(self):
            if self.status_code < 400:
                return True
            return False

        response_mock.__bool__ = bool_mock

        palletjack.utils.requests.get.return_value = response_mock

        result = palletjack.utils.Geocoding.geocode_addr("foo", "bar", "foo_key", (0.015, 0.03))

        assert "No response from GET; request timeout?" not in caplog.text

    def test_geocode_addr_retries_on_exception(self, mocker):
        mocker.patch("palletjack.utils.requests", autospec=True)
        mocker.patch("palletjack.utils.sleep")

        response_mock = mocker.Mock()
        response_mock.json.return_value = {
            "status": 200,
            "result": {"location": {"x": 123, "y": 456}, "score": 100.0, "matchAddress": "bar"},
        }
        response_mock.status_code = 200

        palletjack.utils.requests.get.side_effect = [Exception, response_mock]

        result = palletjack.utils.Geocoding.geocode_addr("foo", "bar", "foo_key", (0.015, 0.03))

        assert palletjack.utils.requests.get.call_count == 2
        assert result == (123, 456, 100.0, "bar")

    def test_geocode_addr_retries_on_missing_response(self, mocker, caplog):
        caplog.set_level(logging.DEBUG)
        mocker.patch("palletjack.utils.requests", autospec=True)
        mocker.patch("palletjack.utils.sleep")

        response_mock = mocker.Mock()
        response_mock.json.return_value = {
            "status": 200,
            "result": {"location": {"x": 123, "y": 456}, "score": 100.0, "matchAddress": "bar"},
        }
        response_mock.status_code = 200

        palletjack.utils.requests.get.side_effect = [None, response_mock]

        result = palletjack.utils.Geocoding.geocode_addr("foo", "bar", "foo_key", (0.015, 0.03))

        assert "No response from GET; request timeout?" in caplog.text
        assert palletjack.utils.requests.get.call_count == 2
        assert result == (123, 456, 100.0, "bar")

    def test_geocode_addr_retries_on_bad_status_code(self, mocker, caplog):
        caplog.set_level(logging.DEBUG)
        mocker.patch("palletjack.utils.requests", autospec=True)
        mocker.patch("palletjack.utils.sleep")

        first_response_mock = mocker.Mock()
        first_response_mock.status_code = 500

        second_response_mock = mocker.Mock()
        second_response_mock.json.return_value = {
            "status": 200,
            "result": {"location": {"x": 123, "y": 456}, "score": 100.0, "matchAddress": "bar"},
        }
        second_response_mock.status_code = 200

        palletjack.utils.requests.get.side_effect = [first_response_mock, second_response_mock]

        result = palletjack.utils.Geocoding.geocode_addr("foo", "bar", "foo_key", (0.015, 0.03))

        assert "Did not receive a valid geocoding response; status code: 500" in caplog.text
        assert palletjack.utils.requests.get.call_count == 2
        assert result == (123, 456, 100.0, "bar")

    def test_geocode_addr_handles_complete_geocode_failure(self, mocker, caplog):
        caplog.set_level(logging.DEBUG)
        mocker.patch("palletjack.utils.requests", autospec=True)
        mocker.patch("palletjack.utils.sleep")

        bad_response = mocker.Mock()
        bad_response.status_code = 500
        palletjack.utils.requests.get.side_effect = [bad_response] * 4

        result = palletjack.utils.Geocoding.geocode_addr("foo", "bar", "foo_key", (0.015, 0.03))

        assert "Did not receive a valid geocoding response; status code: 500" in caplog.messages
        assert palletjack.utils.requests.get.call_count == 4
        assert result == (0, 0, 0.0, "No API response")

    def test_geocode_addr_sleeps_appropriately(self, mocker):
        mocker.patch("palletjack.utils.requests", autospec=True)
        mocker.patch("palletjack.utils.sleep")

        response_mock = mocker.Mock()
        response_mock.json.return_value = {
            "status": 200,
            "result": {"location": {"x": 123, "y": 456}, "score": 100.0, "matchAddress": "bar"},
        }
        response_mock.status_code = 200

        palletjack.utils.requests.get.return_value = response_mock

        palletjack.utils.Geocoding.geocode_addr("foo", "bar", "foo_key", (0.015, 0.03))

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
        req_mock = mocker.patch("palletjack.utils.requests", autospec=True)
        response_mock = mocker.Mock()
        response_mock.json.return_value = {"status": 200, "message": "foo"}
        req_mock.get.return_value = response_mock

        #: Should not raise
        palletjack.utils.Geocoding.validate_api_key("foo")

    def test_validate_api_key_bad_key(self, mocker):
        req_mock = mocker.patch("palletjack.utils.requests", autospec=True)
        response_mock = mocker.Mock()
        response_mock.json.return_value = {"status": 400, "message": "Invalid API key"}
        req_mock.get.return_value = response_mock

        with pytest.raises(ValueError, match=re.escape("API key validation failed: Invalid API key")):
            palletjack.utils.Geocoding.validate_api_key("foo")

    def test_validate_api_key_bad_key_logs(self, mocker, caplog):
        req_mock = mocker.patch("palletjack.utils.requests", autospec=True)
        response_mock = mocker.Mock()
        response_mock.json.return_value = {"status": 400, "message": "Invalid API key"}
        req_mock.get.return_value = response_mock

        with pytest.raises(ValueError, match=re.escape("API key validation failed: Invalid API key")):
            palletjack.utils.Geocoding.validate_api_key("foo")

        caplog.set_level(logging.ERROR, logger="utils")
        assert "API key validation failed: Invalid API key" in caplog.text

    def test_validate_api_key_handles_network_exception(self, mocker, caplog):
        req_mock = mocker.patch("palletjack.utils.requests", autospec=True)
        mocker.patch("palletjack.utils.sleep")
        # caplog.set_level(logging.DEBUG)
        req_mock.get.side_effect = [IOError("Random Error")] * 4

        with pytest.raises(
            RuntimeError,
            match=re.escape("Could not determine key validity; check your API key and/or network connection"),
        ) as exc_info:
            palletjack.utils.Geocoding.validate_api_key("foo")

        assert req_mock.get.call_count == 4
        assert "Random Error" in str(exc_info.value.__cause__)

    def test_validate_api_key_handles_other_response(self, mocker):
        req_mock = mocker.patch("palletjack.utils.requests", autospec=True)
        response_mock = mocker.Mock()
        response_mock.json.return_value = {"status": 404, "message": "Weird Response"}
        req_mock.get.return_value = response_mock

        with pytest.warns(UserWarning, match=re.escape("Unhandled API key validation response 404: Weird Response")):
            palletjack.utils.Geocoding.validate_api_key("foo")


class TestFieldRenaming:
    def test_rename_fields_renames_all_fields(self):
        parcels_df = pd.DataFrame(
            {
                "account_no": [1, 2, 3],
                "type": ["sf", "mf", "condo"],
            }
        )

        field_mapping = {
            "account_no": "PARCEL_ID",
            "type": "class",
        }

        renamed_df = palletjack.utils.rename_fields(parcels_df, field_mapping)

        assert list(renamed_df.columns) == ["PARCEL_ID", "class"]

    def test_rename_fields_renames_some_fields(self):
        parcels_df = pd.DataFrame(
            {
                "account_no": [1, 2, 3],
                "class": ["sf", "mf", "condo"],
            }
        )

        field_mapping = {
            "account_no": "PARCEL_ID",
        }

        renamed_df = palletjack.utils.rename_fields(parcels_df, field_mapping)

        assert list(renamed_df.columns) == ["PARCEL_ID", "class"]

    def test_rename_fields_raises_exception_for_missing_field(self):
        parcels_df = pd.DataFrame(
            {
                "account_no": [1, 2, 3],
                "type": ["sf", "mf", "condo"],
            }
        )

        field_mapping = {
            "account_no": "PARCEL_ID",
            "TYPE": "class",
        }

        with pytest.raises(ValueError) as exception_info:
            renamed_df = palletjack.utils.rename_fields(parcels_df, field_mapping)

            assert "Field TYPE not found in dataframe." in str(exception_info)


class TestAuthorization:
    def test_authorize_pygsheets_auths_from_file(self, mocker):
        pygsheets_mock = mocker.patch.object(palletjack.utils, "pygsheets")
        pygsheets_mock.authorize.return_value = "authed"

        client = palletjack.utils.authorize_pygsheets("file")

        assert pygsheets_mock.authorize.called_once_with("file")
        assert client == "authed"

    def test_authorize_pygsheets_auths_from_custom_credentials_file_not_found(self, mocker, caplog):
        pygsheets_mock = mocker.patch.object(palletjack.utils, "pygsheets")
        pygsheets_mock.authorize.side_effect = [FileNotFoundError, "authed"]

        caplog.set_level(logging.DEBUG, logger="palletjack.utils")
        caplog.clear()

        client = palletjack.utils.authorize_pygsheets("credentials")

        assert "Credentials file not found, trying as environment variable" in [rec.message for rec in caplog.records]

        assert pygsheets_mock.authorize.call_count == 2
        assert pygsheets_mock.authorize.call_args_list == [
            mocker.call(service_file="credentials"),
            mocker.call(custom_credentials="credentials"),
        ]
        assert client == "authed"

    def test_authorize_pygsheets_auths_from_custom_credentials_custom_object(self, mocker, caplog):
        pygsheets_mock = mocker.patch.object(palletjack.utils, "pygsheets")
        pygsheets_mock.authorize.side_effect = [TypeError, "authed"]

        caplog.set_level(logging.DEBUG, logger="palletjack.utils")
        caplog.clear()

        client = palletjack.utils.authorize_pygsheets("credentials")

        assert "Credentials file not found, trying as environment variable" in [rec.message for rec in caplog.records]

        assert pygsheets_mock.authorize.call_count == 2
        assert pygsheets_mock.authorize.call_args_list == [
            mocker.call(service_file="credentials"),
            mocker.call(custom_credentials="credentials"),
        ]
        assert client == "authed"

    def test_authorize_pygsheets_raises_after_failing_both(self, mocker, caplog):
        pygsheets_mock = mocker.patch.object(palletjack.utils, "pygsheets")
        pygsheets_mock.authorize.side_effect = [FileNotFoundError, IOError]

        caplog.set_level(logging.DEBUG, logger="palletjack.utils")
        caplog.clear()

        with pytest.raises(RuntimeError):
            client = palletjack.utils.authorize_pygsheets("credentials")

        assert "Credentials file not found, trying as environment variable" in [rec.message for rec in caplog.records]

        assert pygsheets_mock.authorize.call_count == 2
        assert pygsheets_mock.authorize.call_args_list == [
            mocker.call(service_file="credentials"),
            mocker.call(custom_credentials="credentials"),
        ]


class TestEmptyStringsAsNulls:
    def test_converts_empty_strings_to_null(self):
        FeatureSet = namedtuple("FeatureSet", ["features"])
        Feature = namedtuple("Feature", ["attributes"])
        feature_set = FeatureSet(
            [
                Feature(
                    {
                        "a": "foo",
                        "b": "baz",
                    }
                ),
                Feature(
                    {
                        "a": "",
                        "b": "",
                    }
                ),
            ]
        )
        fields = [
            {
                "name": "a",
                "type": "esriFieldTypeInteger",
                "nullable": False,
            },
            {
                "name": "b",
                "type": "esriFieldTypeInteger",
                "nullable": True,
            },
        ]

        fixed_feature_set = palletjack.utils.fix_numeric_empty_strings(feature_set, fields)
        fixed_feature = fixed_feature_set.features[1]

        assert fixed_feature.attributes["a"] == ""
        assert fixed_feature.attributes["b"] is None

    def test_fix_numeric_empty_strings_handles_both_missing_shape_info_fields(self):
        FeatureSet = namedtuple("FeatureSet", ["features"])
        Feature = namedtuple("Feature", ["attributes"])
        feature_set = FeatureSet(
            [
                Feature(
                    {
                        "a": "foo",
                        "b": "baz",
                    }
                ),
                Feature(
                    {
                        "a": "",
                        "b": "",
                    }
                ),
            ]
        )
        fields = [
            {
                "name": "a",
                "type": "esriFieldTypeInteger",
                "nullable": False,
            },
            {
                "name": "b",
                "type": "esriFieldTypeInteger",
                "nullable": True,
            },
            {
                "name": "Shape__Length",
                "type": "esriFieldTypeDouble",
                "nullable": True,
            },
            {
                "name": "Shape__Area",
                "type": "esriFieldTypeDouble",
                "nullable": True,
            },
        ]

        fixed_feature_set = palletjack.utils.fix_numeric_empty_strings(feature_set, fields)
        fixed_feature = fixed_feature_set.features[1]

        assert fixed_feature.attributes["a"] == ""
        assert fixed_feature.attributes["b"] is None

    def test_fix_numeric_empty_strings_handles_single_missing_shape_info_field(self):
        FeatureSet = namedtuple("FeatureSet", ["features"])
        Feature = namedtuple("Feature", ["attributes"])
        feature_set = FeatureSet(
            [
                Feature(
                    {
                        "a": "foo",
                        "b": "baz",
                    }
                ),
                Feature(
                    {
                        "a": "",
                        "b": "",
                    }
                ),
            ]
        )
        fields = [
            {
                "name": "a",
                "type": "esriFieldTypeInteger",
                "nullable": False,
            },
            {
                "name": "b",
                "type": "esriFieldTypeInteger",
                "nullable": True,
            },
            {
                "name": "Shape__Length",
                "type": "esriFieldTypeDouble",
                "nullable": True,
            },
        ]

        fixed_feature_set = palletjack.utils.fix_numeric_empty_strings(feature_set, fields)
        fixed_feature = fixed_feature_set.features[1]

        assert fixed_feature.attributes["a"] == ""
        assert fixed_feature.attributes["b"] is None


class TestCheckFieldsMatch:
    def test_check_fields_match_normal(self, mocker):
        mock_fl = mocker.Mock()
        mock_fl.properties.fields = [
            {"name": "Foo"},
            {"name": "Bar"},
        ]
        df = pd.DataFrame(columns=["Foo", "Bar"])

        palletjack.utils.check_fields_match(mock_fl, df)

    def test_check_fields_match_raises_error_on_extra_new_field(self, mocker):
        mock_fl = mocker.Mock()
        mock_fl.properties.fields = [
            {"name": "Foo"},
            {"name": "Bar"},
        ]
        df = pd.DataFrame(columns=["Foo", "Bar", "Baz"])

        with pytest.raises(RuntimeError) as exc_info:
            palletjack.utils.check_fields_match(mock_fl, df)

        assert (
            exc_info.value.args[0]
            == "New dataset contains the following fields that are not present in the live dataset: {'Baz'}"
        )

    def test_check_fields_match_ignores_new_shape_field(self, mocker):
        mock_fl = mocker.Mock()
        mock_fl.properties.fields = [
            {"name": "Foo"},
            {"name": "Bar"},
        ]
        df = pd.DataFrame(columns=["Foo", "Bar", "SHAPE"])

        palletjack.utils.check_fields_match(mock_fl, df)

    def test_check_fields_match_warns_on_missing_new_field(self, mocker, caplog):
        mock_fl = mocker.Mock()
        mock_fl.properties.fields = [
            {"name": "Foo"},
            {"name": "Bar"},
            {"name": "Baz"},
        ]
        df = pd.DataFrame(columns=["Foo", "Bar"])

        palletjack.utils.check_fields_match(mock_fl, df)

        assert (
            "New dataset does not contain the following fields that are present in the live dataset: {'Baz'}"
            in caplog.text
        )

    def test_check_live_and_new_field_types_match_normal(self, mocker):
        new_df = pd.DataFrame(
            {
                "ints": [1, 2, 3],
                "floats": [4.0, 5.0, 6.0],
                "strings": ["a", "b", "c"],
                "OBJECTID": [11, 12, 13],
                "GlobalID": [
                    "cc1cd617-1e55-4153-914d-8abb6ef22f24",
                    "0f45d56f-249e-494a-863e-6b3999619bae",
                    "d3a64873-8a09-4351-9ea0-802e450329ea",
                ],
                "dates": ["2015-09-02T23:08:12+00:00", "2015-09-02T23:08:13+00:00", "2015-09-02T23:08:14+00:00"],
            }
        )
        new_df["datetimes"] = pd.to_datetime(new_df["dates"]).dt.tz_localize(None)
        properties_mock = mocker.Mock()
        properties_mock.fields = [
            {"name": "OBJECTID", "type": "esriFieldTypeOID"},
            {"name": "strings", "type": "esriFieldTypeString"},
            {"name": "ints", "type": "esriFieldTypeInteger"},
            {"name": "floats", "type": "esriFieldTypeDouble"},
            {"name": "GlobalID", "type": "esriFieldTypeGlobalID"},
            {"name": "datetimes", "type": "esriFieldTypeDate"},
        ]

        #: If it raises an error, it failed.
        checker = palletjack.utils.FieldChecker(properties_mock, new_df)
        checker.check_live_and_new_field_types_match(["ints", "floats", "strings", "OBJECTID", "GlobalID", "datetimes"])

    def test_check_live_and_new_field_types_match_converted(self, mocker):
        new_df = pd.DataFrame(
            {
                "ints": [1, 2, 3],
                "floats": [4.1, 5.1, 6.1],
                "strings": ["a", "b", "c"],
                "OBJECTID": [11, 12, 13],
                "GlobalID": [
                    "cc1cd617-1e55-4153-914d-8abb6ef22f24",
                    "0f45d56f-249e-494a-863e-6b3999619bae",
                    "d3a64873-8a09-4351-9ea0-802e450329ea",
                ],
                "dates": ["2015-09-02T23:08:12+00:00", "2015-09-02T23:08:13+00:00", "2015-09-02T23:08:14+00:00"],
            }
        ).convert_dtypes()

        new_df["datetimes"] = pd.to_datetime(new_df["dates"]).dt.tz_localize(None)

        properties_mock = mocker.Mock()
        properties_mock.fields = [
            {"name": "OBJECTID", "type": "esriFieldTypeOID"},
            {"name": "strings", "type": "esriFieldTypeString"},
            {"name": "ints", "type": "esriFieldTypeInteger"},
            {"name": "floats", "type": "esriFieldTypeDouble"},
            {"name": "GlobalID", "type": "esriFieldTypeGlobalID"},
            {"name": "datetimes", "type": "esriFieldTypeDate"},
        ]

        #: If it raises an error, it failed.
        checker = palletjack.utils.FieldChecker(properties_mock, new_df)
        checker.check_live_and_new_field_types_match(["ints", "floats", "strings", "OBJECTID", "GlobalID", "datetimes"])

    def test_check_live_and_new_field_types_match_raises_on_incompatible(self, mocker):
        new_df = pd.DataFrame(
            {
                "ints": [1, 2, 3],
            }
        )

        properties_mock = mocker.Mock()
        properties_mock.fields = [{"name": "ints", "type": "esriFieldTypeDouble"}]

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Field type incompatibilities (field, live type, new type): [('ints', 'esriFieldTypeDouble', 'int64')]"
            ),
        ):
            checker = palletjack.utils.FieldChecker(properties_mock, new_df)
            checker.check_live_and_new_field_types_match(["ints"])

    def test_check_live_and_new_field_types_match_handles_nullable_int_with_nans(self, mocker):
        new_df = pd.DataFrame(
            {
                "ints": [1, 2, None],
            }
        ).convert_dtypes()

        properties_mock = mocker.Mock()
        properties_mock.fields = [{"name": "ints", "type": "esriFieldTypeInteger"}]

        checker = palletjack.utils.FieldChecker(properties_mock, new_df)

        #: should not raise
        checker.check_live_and_new_field_types_match(["ints"])

    def test_check_live_and_new_field_types_match_raises_on_multiple_incompatible(self, mocker):
        new_df = pd.DataFrame(
            {
                "ints": [1, 2, 3],
                "floats": [1.1, 1.2, 1.3],
            }
        )
        properties_mock = mocker.Mock()
        properties_mock.fields = [
            {"name": "ints", "type": "esriFieldTypeDouble"},
            {"name": "floats", "type": "esriFieldTypeString"},
        ]

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Field type incompatibilities (field, live type, new type): [('ints', 'esriFieldTypeDouble', 'int64'), ('floats', 'esriFieldTypeString', 'float64')]"
            ),
        ):
            checker = palletjack.utils.FieldChecker(properties_mock, new_df)
            checker.check_live_and_new_field_types_match(["ints", "floats"])

    def test_check_live_and_new_field_types_match_raises_on_incoming_int_as_float(self, mocker):
        new_df = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [1.1, 1.2, np.nan],
            }
        )
        properties_mock = mocker.Mock()
        properties_mock.fields = [
            {"name": "a", "type": "esriFieldTypeDouble"},
            {"name": "b", "type": "esriFieldTypeInteger"},
        ]

        with pytest.raises(
            palletjack.IntFieldAsFloatError,
            match=re.escape(
                "Field type incompatibilities (field, live type, new type): [('a', 'esriFieldTypeDouble', 'int64'), ('b', 'esriFieldTypeInteger', 'float64')]\nCheck the following int fields for null/np.nan values and convert to panda's nullable int dtype: b"
            ),
        ):
            checker = palletjack.utils.FieldChecker(properties_mock, new_df)
            checker.check_live_and_new_field_types_match(["a", "b"])

    def test_check_live_and_new_field_types_match_raises_on_timezone_aware_datetime(self, mocker):
        new_df = pd.DataFrame(
            {
                "a": ["2020-04-29T01:09:29+00:00"],
            }
        )
        new_df["a"] = pd.to_datetime(new_df["a"])
        properties_mock = mocker.Mock()
        properties_mock.fields = [
            {"name": "a", "type": "esriFieldTypeDate"},
        ]

        with pytest.raises(
            palletjack.TimezoneAwareDatetimeError,
            match=re.escape(
                "Field type incompatibilities (field, live type, new type): [('a', 'esriFieldTypeDate', 'datetime64[ns, UTC]')]\nCheck the following datetime fields for timezone aware dtypes values and convert to timezone-naive dtypes using pd.to_datetime(df['field']).dt.tz_localize(None): a"
            ),
        ):
            checker = palletjack.utils.FieldChecker(properties_mock, new_df)
            checker.check_live_and_new_field_types_match(["a"])

    def test_check_live_and_new_field_types_match_raises_on_notimplemented_esri_type(self, mocker):
        new_df = pd.DataFrame(
            {
                "ints": [1, 2, 3],
            }
        )

        properties_mock = mocker.Mock()
        properties_mock.fields = [{"name": "ints", "type": "esriFieldTypeXML"}]

        with pytest.raises(NotImplementedError) as exc_info:
            checker = palletjack.utils.FieldChecker(properties_mock, new_df)
            checker.check_live_and_new_field_types_match(["ints"])

        assert 'Live field "ints" type "esriFieldTypeXML" not yet mapped to a pandas dtype' in str(exc_info.value)

    def test_check_live_and_new_field_types_removes_SHAPE(self, mocker):
        geocheck_mock = mocker.patch("palletjack.utils.FieldChecker._check_geometry_types")
        new_df = pd.DataFrame(
            {
                "ints": [1, 2, 3],
                "SHAPE": [geometry.Geometry([0, 0])] * 3,
            }
        ).convert_dtypes()

        properties_mock = mocker.Mock()
        properties_mock.fields = [{"name": "ints", "type": "esriFieldTypeInteger"}]

        #: If it raises an error, it failed.
        checker = palletjack.utils.FieldChecker(properties_mock, new_df)
        checker.check_live_and_new_field_types_match(["ints", "SHAPE"])
        geocheck_mock.assert_called_once()

    def test_check_geometry_types_normal(self, mocker):
        new_df = pd.DataFrame.spatial.from_xy(
            pd.DataFrame(
                {
                    "OBJECTID": [11, 12, 13],
                    "x": [0, 0, 0],
                    "y": [0, 0, 0],
                }
            ),
            x_column="x",
            y_column="y",
        )

        properties_mock = mocker.Mock()
        properties_mock.geometryType = "esriGeometryPoint"
        properties_mock.fields = {"a": ["b"]}

        #: If it raises an error, it failed.
        checker = palletjack.utils.FieldChecker(properties_mock, new_df)
        checker._check_geometry_types()

    def test_check_geometry_types_raises_on_multiple_types(self, mocker):
        new_df = mocker.MagicMock()
        new_df.columns = ["SHAPE"]
        new_df.spatial.geometry_type = [1, 2]
        new_df["SHAPE"].isna.return_value.any.return_value = False

        properties_mock = mocker.Mock()
        properties_mock.geometryType = "esriGeometryPoint"
        properties_mock.fields = {"a": ["b"]}

        with pytest.raises(ValueError) as exc_info:
            checker = palletjack.utils.FieldChecker(properties_mock, new_df)
            checker._check_geometry_types()

        assert "New dataframe has multiple geometry types" in str(exc_info.value)

    def test_check_geometry_types_raises_on_incompatible_type(self, mocker):
        new_df = mocker.MagicMock()
        new_df.columns = ["SHAPE"]
        new_df.spatial.geometry_type = ["Polygon"]
        new_df["SHAPE"].isna.return_value.any.return_value = False

        properties_mock = mocker.Mock()
        properties_mock.geometryType = "esriGeometryPoint"
        properties_mock.fields = {"a": ["b"]}

        with pytest.raises(ValueError) as exc_info:
            checker = palletjack.utils.FieldChecker(properties_mock, new_df)
            checker._check_geometry_types()

        assert 'New dataframe geometry type "Polygon" incompatible with live geometry type "esriGeometryPoint"' in str(
            exc_info.value
        )

    def test_check_geometry_types_raises_on_missing_SHAPE(self, mocker):
        new_df = mocker.Mock()
        new_df.columns = ["foo"]

        checker_mock = mocker.Mock()
        checker_mock.new_dataframe = new_df

        with pytest.raises(ValueError) as exc_info:
            palletjack.utils.FieldChecker._check_geometry_types(checker_mock)

        assert "New dataframe does not have a SHAPE column" in str(exc_info.value)

    def test_check_geometry_types_raises_on_missing_geometry_type(self, mocker):
        new_df = pd.DataFrame({"SHAPE": ["polyline", None]})

        properties_mock = mocker.Mock()
        properties_mock.geometryType = "esriGeometryPolyline"
        properties_mock.fields = {"a": ["b"]}

        with pytest.raises(ValueError) as exc_info:
            checker = palletjack.utils.FieldChecker(properties_mock, new_df)
            checker._check_geometry_types()

        assert "New dataframe has missing geometries at index [1]" in str(exc_info.value)


class TestCheckGeometryTypes:
    def test_check_geometry_types_handles_geodataframe(self, mocker):
        self_mock = mocker.Mock()

        new_df = mocker.MagicMock(spec="palletjack.utils.pd.DataFrame")
        new_df.columns = ["SHAPE"]
        new_df["SHAPE"].isna.return_value.any.return_value = False
        new_df.geom_type = pd.Series(["MultiPolygon", "MultiPolygon", "MultiPolygon"])

        self_mock.new_dataframe = new_df
        self_mock.live_data_properties.geometryType = "esriGeometryPolygon"
        self_mock._condense_geopandas_multi_types.return_value = np.array(["MultiPolygon"])

        #: If it raises an error, it failed.
        palletjack.utils.FieldChecker._check_geometry_types(self_mock)

        self_mock._condense_geopandas_multi_types.assert_called_once_with(np.array(["MultiPolygon"]))

    def test_check_geometry_types_raises_on_multiple_gdf_types(self, mocker):
        self_mock = mocker.Mock()

        new_df = mocker.MagicMock(spec="palletjack.utils.pd.DataFrame")
        new_df.columns = ["SHAPE"]
        new_df["SHAPE"].isna.return_value.any.return_value = False
        new_df.geom_type = pd.Series(["Point", "Polygon"])

        self_mock.new_dataframe = new_df
        self_mock.live_data_properties.geometryType = "esriGeometryPoint"
        self_mock._condense_geopandas_multi_types.return_value = np.array(["Point", "Polygon"])

        with pytest.raises(ValueError) as exc_info:
            palletjack.utils.FieldChecker._check_geometry_types(self_mock)

        assert "New dataframe has multiple geometry types" in str(exc_info.value)


class TestCondenseGeopandasMultiTypes:
    def test__condense_geopandas_multi_types_single_type_passes_through(self, mocker):
        unique_types = np.array(["Polygon"])
        result = palletjack.utils.FieldChecker._condense_geopandas_multi_types(mocker.Mock(), unique_types)
        np.testing.assert_array_equal(result, ["Polygon"])

    def test__condense_geopandas_multi_types_polygon_and_multipolygon_condenses_to_multipolygon(self, mocker):
        unique_types = np.array(["Polygon", "MultiPolygon"])
        result = palletjack.utils.FieldChecker._condense_geopandas_multi_types(mocker.Mock(), unique_types)

        np.testing.assert_array_equal(result, ["MultiPolygon"])

    def test__condense_geopandas_multi_types_linestring_and_multilinestring_condenses_to_multilinestring(self, mocker):
        unique_types = np.array(["LineString", "MultiLineString"])
        result = palletjack.utils.FieldChecker._condense_geopandas_multi_types(mocker.Mock(), unique_types)

        np.testing.assert_array_equal(result, ["MultiLineString"])

    def test__condense_geopandas_multi_types_point_and_multipoint_does_not_condense(self, mocker):
        unique_types = np.array(["Point", "MultiPoint"])
        result = palletjack.utils.FieldChecker._condense_geopandas_multi_types(mocker.Mock(), unique_types)

        np.testing.assert_array_equal(result, ["Point", "MultiPoint"])

    def test__condense_geopandas_multi_types_passes_heterogenous_types(self, mocker):
        unique_types = np.array(["LineString", "MultiLineString", "Polygon"])
        result = palletjack.utils.FieldChecker._condense_geopandas_multi_types(mocker.Mock(), unique_types)

        np.testing.assert_array_equal(result, ["MultiLineString", "Polygon"])


class TestNullableIntWarning:
    def test_check_nullable_ints_shapely_warns_with_na(self, mocker):
        new_df = pd.DataFrame(
            {
                "normal": [1, 2, 3],
                "nulls": [4, 5, None],
                "floats": [6.1, 7.1, 8.1],
            }
        ).convert_dtypes()

        mocker.patch("palletjack.utils.importlib.util.find_spec", return_value=None)

        checker_mock = mocker.Mock()
        checker_mock.new_dataframe = new_df

        with pytest.warns(
            UserWarning,
            match=re.escape(
                "The following columns have null values that will be replaced by 0 due to shapely conventions: nulls"
            ),
        ):
            palletjack.utils.FieldChecker.check_nullable_ints_shapely(checker_mock)

    def test_check_nullable_ints_shapely_doesnt_warn_on_float_nan(self, mocker):
        new_df = pd.DataFrame(
            {
                "normal": [1, 2, 3],
                "nulls": [4, 5, 6],
                "floats": [None, 7.1, 8.1],
            }
        ).convert_dtypes()

        mocker.patch("palletjack.utils.importlib.util.find_spec", return_value=None)

        checker_mock = mocker.Mock()
        checker_mock.new_dataframe = new_df
        warning_mock = mocker.patch("palletjack.utils.warnings")

        palletjack.utils.FieldChecker.check_nullable_ints_shapely(checker_mock)

        assert warning_mock.warns.call_count == 0

    def test_check_nullable_ints_shapely_doesnt_warn_when_using_arcpy(self, mocker):
        new_df = pd.DataFrame(
            {
                "normal": [1, 2, 3],
                "nulls": [4, 5, None],
                "floats": [6.1, 7.1, 8.1],
            }
        ).convert_dtypes()

        mocker.patch("palletjack.utils.importlib.util.find_spec", return_value="__spec__")

        checker_mock = mocker.Mock()
        checker_mock.new_dataframe = new_df
        warning_mock = mocker.patch("palletjack.utils.warnings")

        palletjack.utils.FieldChecker.check_nullable_ints_shapely(checker_mock)

        assert warning_mock.warns.call_count == 0


class TestFieldNullChecker:
    def test_check_for_non_null_fields_raises_on_null_data_in_nonnullable_field(self, mocker):
        properties_mock = mocker.Mock()
        properties_mock.fields = [
            {
                "name": "regular",
                "nullable": True,
                "defaultValue": None,
            },
            {
                "name": "non-nullable",
                "nullable": False,
                "defaultValue": None,
            },
        ]

        new_df = pd.DataFrame(
            {
                "regular": ["a", "b"],
                "non-nullable": ["c", None],
            }
        )

        checker = palletjack.utils.FieldChecker(properties_mock, new_df)

        with pytest.raises(ValueError) as exc_info:
            checker.check_for_non_null_fields(["regular", "non-nullable"])

        assert (
            "The following fields cannot have null values in the live data but one or more nulls exist in the new data: non-nullable"
            in str(exc_info.value)
        )

    def test_check_for_non_null_fields_doesnt_raise_on_null_in_nullable_field(self, mocker):
        properties_mock = mocker.Mock()
        properties_mock.fields = [
            {
                "name": "regular",
                "nullable": True,
                "defaultValue": None,
            },
        ]

        new_df = pd.DataFrame(
            {
                "regular": ["a", None],
            }
        )

        checker = palletjack.utils.FieldChecker(properties_mock, new_df)

        #: Should not raise an error
        checker.check_for_non_null_fields(["regular"])

    def test_check_for_non_null_fields_doesnt_raise_on_null_in_nonnullable_with_default(self, mocker):
        properties_mock = mocker.Mock()
        properties_mock.fields = [
            {
                "name": "regular",
                "nullable": True,
                "defaultValue": "foo",
            },
        ]

        new_df = pd.DataFrame(
            {
                "regular": ["a", None],
            }
        )

        checker = palletjack.utils.FieldChecker(properties_mock, new_df)

        #: Should not raise an error
        checker.check_for_non_null_fields(["regular"])

    def test_check_for_non_null_fields_skips_field(self, mocker):
        properties_mock = mocker.Mock()
        properties_mock.fields = [
            {
                "name": "regular",
                "nullable": True,
                "defaultValue": None,
            },
            {
                "name": "non-nullable",
                "nullable": False,
                "defaultValue": None,
            },
        ]

        new_df = pd.DataFrame(
            {
                "regular": ["a", "b"],
                "non-nullable": ["c", None],
            }
        )

        checker = palletjack.utils.FieldChecker(properties_mock, new_df)

        #: Should not raise
        checker.check_for_non_null_fields(["regular"])


class TestFieldLength:
    def test_check_field_length_normal_string(self, mocker):
        properties_mock = mocker.Mock()
        properties_mock.fields = [
            {
                "name": "foo",
                "type": "esriFieldTypeString",
                "length": 10,
            },
        ]

        new_df = pd.DataFrame(
            {
                "foo": ["aaa", "bbbb"],
            }
        )

        checker = palletjack.utils.FieldChecker(properties_mock, new_df)

        #: Should not raise
        checker.check_field_length(["foo"])

    def test_check_field_length_can_handle_all_null_strings(self, mocker):
        properties_mock = mocker.Mock()
        properties_mock.fields = [
            {
                "name": "foo",
                "type": "esriFieldTypeString",
                "length": 10,
            },
        ]

        new_df = pd.DataFrame(
            {
                "foo": pd.Series([None, None], dtype="string"),
            }
        )

        checker = palletjack.utils.FieldChecker(properties_mock, new_df)

        #: Should not raise
        checker.check_field_length(["foo"])

    def test_check_field_length_raises_on_long_string(self, mocker):
        properties_mock = mocker.Mock()
        properties_mock.fields = [
            {
                "name": "foo",
                "type": "esriFieldTypeString",
                "length": 10,
            },
        ]

        new_df = pd.DataFrame(
            {
                "foo": ["aaa", "bbbb", "this string is far too long"],
            }
        )

        checker = palletjack.utils.FieldChecker(properties_mock, new_df)

        with pytest.raises(ValueError) as exc_info:
            checker.check_field_length(["foo"])

        assert "Row 2, column foo in new data exceeds the live data max length of 10" in str(exc_info.value)

    def test_check_field_length_uses_fields_arg(self, mocker):
        properties_mock = mocker.Mock()
        properties_mock.fields = [
            {
                "name": "foo",
                "type": "esriFieldTypeString",
                "length": 10,
            },
            {
                "name": "bar",
                "type": "esriFieldTypeString",
                "length": 10,
            },
        ]
        new_df = pd.DataFrame(
            {
                "foo": ["aaa", "bbbb"],
                "bar": ["a", "way too long field"],
            }
        )

        checker = palletjack.utils.FieldChecker(properties_mock, new_df)

        #: bar shouldn't trigger an exception
        checker.check_field_length(["foo"])

    def test_check_field_length_uses_ignores_new_field_not_in_live_data(self, mocker):
        properties_mock = mocker.Mock()
        properties_mock.fields = [
            {
                "name": "foo",
                "type": "esriFieldTypeString",
                "length": 10,
            },
        ]
        new_df = pd.DataFrame(
            {
                "foo": ["aaa", "bbbb"],
                "bar": ["a", "way too long field"],
            }
        )

        checker = palletjack.utils.FieldChecker(properties_mock, new_df)

        #: bar shouldn't trigger an exception
        checker.check_field_length(["foo"])

    def test_check_field_length_works_with_int_field(self, mocker):
        properties_mock = mocker.Mock()
        properties_mock.fields = [
            {
                "name": "foo",
                "type": "esriFieldTypeString",
                "length": 10,
            },
            {
                "name": "int",
                "type": "esriFieldTypeInteger",
            },
        ]
        new_df = pd.DataFrame(
            {
                "foo": ["aaa", "bbbb"],
                "int": [1, 2],
            }
        )

        checker = palletjack.utils.FieldChecker(properties_mock, new_df)

        #: Should not raise
        checker.check_field_length(["foo", "int"])

    def test_check_field_length_passes_with_only_int_fields(self, mocker, caplog):
        caplog.set_level(logging.DEBUG)
        properties_mock = mocker.Mock()
        properties_mock.fields = [
            {
                "name": "foo",
                "type": "esriFieldTypeInteger",
            },
            {
                "name": "int",
                "type": "esriFieldTypeInteger",
            },
        ]
        new_df = pd.DataFrame(
            {
                "foo": ["aaa", "bbbb"],
                "int": [1, 2],
            }
        )

        checker = palletjack.utils.FieldChecker(properties_mock, new_df)

        #: Should not raise
        checker.check_field_length(["foo"])

        assert "No fields with length property" in caplog.text


class TestFieldsPresent:
    def test_check_fields_present_normal_in_both(self, mocker):
        properties_mock = mocker.Mock()
        properties_mock.fields = [{"name": "foo"}, {"name": "bar"}]

        new_df = pd.DataFrame(columns=["foo", "bar"])

        checker = palletjack.utils.FieldChecker(properties_mock, new_df)

        #: Should not raise
        checker.check_fields_present(["foo", "bar"], False)

    def test_check_fields_present_raises_missing_live(self, mocker):
        properties_mock = mocker.Mock()
        properties_mock.fields = [{"name": "foo"}]

        new_df = pd.DataFrame(columns=["foo", "bar"])

        checker = palletjack.utils.FieldChecker(properties_mock, new_df)

        with pytest.raises(RuntimeError) as exc_info:
            checker.check_fields_present(["foo", "bar"], False)
        assert "Fields missing in live data: bar" in str(exc_info.value)

    def test_check_fields_present_raises_missing_new(self, mocker):
        properties_mock = mocker.Mock()
        properties_mock.fields = [{"name": "foo"}, {"name": "bar"}]

        new_df = pd.DataFrame(columns=["foo"])

        checker = palletjack.utils.FieldChecker(properties_mock, new_df)

        with pytest.raises(RuntimeError) as exc_info:
            checker.check_fields_present(["foo", "bar"], False)
        assert "Fields missing in new data: bar" in str(exc_info.value)

    def test_check_fields_present_raises_missing_both(self, mocker):
        properties_mock = mocker.Mock()
        properties_mock.fields = [{"name": "foo"}]

        new_df = pd.DataFrame(columns=["bar"])

        checker = palletjack.utils.FieldChecker(properties_mock, new_df)

        with pytest.raises(RuntimeError) as exc_info:
            checker.check_fields_present(["foo", "bar"], False)
        assert "Fields missing in live data: bar. Fields missing in new data: foo" in str(exc_info.value)

    def test_check_fields_present_adds_oid_to_list_of_fields_to_check(self, mocker):
        properties_mock = mocker.Mock()
        properties_mock.fields = [{"name": "foo"}, {"name": "bar"}, {"name": "OBJECTID"}]

        new_df = pd.DataFrame(columns=["foo", "bar", "OBJECTID"])

        checker = palletjack.utils.FieldChecker(properties_mock, new_df)

        #: Should not raise
        checker.check_fields_present(["foo", "bar"], True)


class TestSRSCheck:
    def test_check_srs_wgs84_good_match(self, mocker):
        checker_mock = mocker.Mock()
        checker_mock.new_dataframe.spatial.sr.wkid = 4326

        #: Should not raise
        palletjack.utils.FieldChecker.check_srs_wgs84(checker_mock)

    def test_check_srs_wgs84_raises_on_mismatch(self, mocker):
        checker_mock = mocker.Mock()
        checker_mock.new_dataframe.spatial.sr.wkid = 42

        with pytest.raises(
            ValueError,
            match=re.escape("New dataframe SRS 42 is not wkid 4326. Reproject with appropriate transformation"),
        ):
            palletjack.utils.FieldChecker.check_srs_wgs84(checker_mock)

    def test_check_srs_wgs84_handles_string_and_int(self, mocker):
        checker_mock = mocker.Mock()
        checker_mock.new_dataframe.spatial.sr.wkid = "4326"

        #: should not raise
        palletjack.utils.FieldChecker.check_srs_wgs84(checker_mock)

    def test_check_srs_wgs84_reports_uncastable_string(self, mocker):
        checker_mock = mocker.Mock()
        checker_mock.new_dataframe.spatial.sr.wkid = "forty two"

        with pytest.raises(ValueError, match=re.escape("Could not cast new SRS to int")) as exc_info:
            palletjack.utils.FieldChecker.check_srs_wgs84(checker_mock)

        assert hasattr(exc_info.value, "__cause__")
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert "invalid literal for int() with base 10:" in str(exc_info.value.__cause__)

    def test_check_srs_wgs84_handles_srs_with_key_of_0(self, mocker):
        checker_mock = mocker.Mock()
        checker_mock.new_dataframe.spatial.sr = {0: 4326}

        #: Should not raise
        palletjack.utils.FieldChecker.check_srs_wgs84(checker_mock)


class TestNullGeometryGenerators:
    def test_get_null_geometries_point(self, mocker):
        properties_mock = mocker.Mock()
        properties_mock.geometryType = "esriGeometryPoint"

        nullo = palletjack.utils.get_null_geometries(properties_mock)

        assert nullo == '{"x": 0, "y": 0, "spatialReference": {"wkid": 4326}}'

    def test_get_null_geometries_polyline(self, mocker):
        properties_mock = mocker.Mock()
        properties_mock.geometryType = "esriGeometryPolyline"

        nullo = palletjack.utils.get_null_geometries(properties_mock)

        assert nullo == '{"paths": [[[0, 0], [0.1, 0.1], [0.2, 0.2]]], "spatialReference": {"wkid": 4326}}'

    def test_get_null_geometries_polygon(self, mocker):
        properties_mock = mocker.Mock()
        properties_mock.geometryType = "esriGeometryPolygon"

        nullo = palletjack.utils.get_null_geometries(properties_mock)

        assert nullo == '{"rings": [[[0, 0.1], [0.1, 0.1], [0.1, 0], [0, 0]]], "spatialReference": {"wkid": 4326}}'

    def test_get_null_geometries_raises_on_other(self, mocker):
        properties_mock = mocker.Mock()
        properties_mock.geometryType = "other"

        with pytest.raises(NotImplementedError) as exc_info:
            nullo = palletjack.utils.get_null_geometries(properties_mock)

        assert "Null value generator for live geometry type other not yet implemented" in str(exc_info.value)


class TestDeleteUtils:
    def test_check_delete_oids_are_ints_casts_string_list_with_spaces(self):
        oid_list = [" 1", " 2", "3"]

        numeric_oids = palletjack.utils.DeleteUtils.check_delete_oids_are_ints(oid_list)

        assert numeric_oids == [1, 2, 3]

    def test_check_delete_oids_are_ints_casts_string_list_without_spaces(self):
        oid_list = ["1", "2", "3"]

        numeric_oids = palletjack.utils.DeleteUtils.check_delete_oids_are_ints(oid_list)

        assert numeric_oids == [1, 2, 3]

    def test_check_delete_oids_are_ints_passes_on_int_list(self):
        oid_list = [1, 2, 3]

        numeric_oids = palletjack.utils.DeleteUtils.check_delete_oids_are_ints(oid_list)

        assert numeric_oids == [1, 2, 3]

    def test_check_delete_oids_are_ints_raises_first_non_int(self):
        oid_list = ["one", 2, 3]

        with pytest.raises(TypeError) as exc_info:
            palletjack.utils.DeleteUtils.check_delete_oids_are_ints(oid_list)

        assert "Couldn't convert OBJECTID(s) `['one']` to integer" in str(exc_info.value)

    def test_check_delete_oids_are_ints_raises_later_non_ints(self):
        oid_list = [1, "two", "three"]

        with pytest.raises(TypeError) as exc_info:
            palletjack.utils.DeleteUtils.check_delete_oids_are_ints(oid_list)

        assert "Couldn't convert OBJECTID(s) `['two', 'three']` to integer" in str(exc_info.value)

    def test_check_for_empty_oid_list_doesnt_raise_on_list(self):
        oid_list = [1, 2, 3]
        numeric_oids = [1, 2, 3]

        palletjack.utils.DeleteUtils.check_for_empty_oid_list(oid_list, numeric_oids)

    def test_check_for_empty_oid_list_raises_on_empty_list(self):
        oid_list = []
        numeric_oids = []

        with pytest.raises(ValueError) as exc_info:
            palletjack.utils.DeleteUtils.check_for_empty_oid_list(oid_list, numeric_oids)

        assert "No OBJECTIDs found in []" in str(exc_info.value)

    def test_check_delete_oids_are_in_live_data_doesnt_warn_on_good_oids(self, mocker):
        fl_mock = mocker.Mock()
        fl_mock.query.return_value = {"objectIdFieldName": "OBJECTID", "objectIds": [1, 2, 3]}
        oid_list = [1, 2, 3]
        oid_numeric = [1, 2, 3]

        palletjack.utils.DeleteUtils.check_delete_oids_are_in_live_data(oid_list, oid_numeric, fl_mock)

    def test_check_delete_oids_are_in_live_data_warns_on_missing_oid(self, mocker):
        fl_mock = mocker.Mock()
        fl_mock.query.return_value = {"objectIdFieldName": "OBJECTID", "objectIds": [1, 2]}
        oid_list = [1, 2, 3]
        oid_numeric = [1, 2, 3]

        with pytest.warns(UserWarning, match=re.escape("OBJECTIDs {3} were not found in the live data")):
            number_of_missing_oids = palletjack.utils.DeleteUtils.check_delete_oids_are_in_live_data(
                oid_list, oid_numeric, fl_mock
            )

        assert number_of_missing_oids == 1


class TestSaveDataframeToGDF:
    def test_save_to_gdb_calls_to_file_with_right_path(self, mocker):
        expected_out_path = Path("foo", "backup.gdb")
        expected_out_layer = f"flayer_{datetime.date.today().strftime('%Y_%m_%d')}"

        mock_fl = mocker.Mock(spec=FeatureLayer)
        mock_fl.properties.name = "flayer"
        mock_fl.properties.type = "Feature Layer"
        mock_fl.query.return_value.sdf.empty = False
        gdf_mock = mocker.patch("palletjack.utils.sedf_to_gdf").return_value

        out_path = palletjack.utils.save_to_gdb(mock_fl, "foo")

        assert out_path == expected_out_path
        gdf_mock.to_file.assert_called_once_with(
            expected_out_path, layer=expected_out_layer, engine="pyogrio", driver="OpenFileGDB"
        )

    def test_save_to_gdb_uses_gdb_for_tables(self, mocker):
        expected_out_path = Path("foo", "backup.gdb")
        expected_out_layer = f"table_{datetime.date.today().strftime('%Y_%m_%d')}"

        mock_tb = mocker.Mock(spec=Table)
        mock_tb.properties.name = "table"
        mock_tb.properties.type = "Table"
        mock_tb.query.return_value.sdf.empty = False
        gdf_mock = mocker.patch("palletjack.utils.gpd.GeoDataFrame").return_value

        out_path = palletjack.utils.save_to_gdb(mock_tb, "foo")

        assert out_path == expected_out_path
        gdf_mock.to_file.assert_called_once()

    def test_save_to_gdb_doesnt_save_empty_data(self, mocker):
        mock_fl = mocker.Mock()
        mock_fl.properties.name = "flayer"
        mock_fl.query.return_value.sdf.empty = True
        gdf_mock = mocker.patch("palletjack.utils.sedf_to_gdf").return_value

        out_path = palletjack.utils.save_to_gdb(mock_fl, "foo")

        gdf_mock.to_file.assert_not_called()
        assert out_path == "No data to save in feature layer flayer"

    def test_save_to_gdb_raises_on_gdb_write_error(self, mocker):
        gdb_path = Path("/foo/bar/backup.gdb")
        date = datetime.date.today().strftime("%Y_%m_%d")
        expected_error = f"Error writing flayer_{date} to {gdb_path}. Verify {gdb_path.parent} exists and is writable."

        mock_fl = mocker.Mock(spec=FeatureLayer)
        mock_fl.properties.name = "flayer"
        mock_fl.properties.type = "Feature Layer"
        mock_fl.query.return_value.sdf.empty = False
        gdf_mock = mocker.patch("palletjack.utils.sedf_to_gdf").return_value
        gdf_mock.to_file.side_effect = pyogrio.errors.DataSourceError

        with pytest.raises(ValueError, match=re.escape(expected_error)):
            out_path = palletjack.utils.save_to_gdb(mock_fl, "/foo/bar")


class TestDataFrameChunking:
    def test_ceildiv_always_returns_at_least_one(self):
        assert palletjack.utils.Chunking._ceildiv(1, 2) == 1

    def test_ceildiv_returns_ceiling(self):
        assert palletjack.utils.Chunking._ceildiv(3, 2) == 2

    def test_chunk_dataframe_properly_chunks(self, iris):
        chunks = 4
        chunk_size = palletjack.utils.Chunking._ceildiv(len(iris), chunks)
        dfs = palletjack.utils.Chunking._chunk_dataframe(iris, chunk_size)

        assert len(dfs) == chunks
        assert [len(df) for df in dfs] == [38, 38, 38, 36]

    def test_chunk_dataframe_properly_chunks_even_sized_chunks(self, iris):
        chunks = 5
        chunk_size = palletjack.utils.Chunking._ceildiv(len(iris), chunks)
        dfs = palletjack.utils.Chunking._chunk_dataframe(iris, chunk_size)

        assert len(dfs) == chunks
        assert [len(df) for df in dfs] == [30] * chunks

    def test_chunk_dataframe_raises_on_single_row_chunk(self, mocker):
        df = pd.DataFrame(["a"], columns=["foo"])

        with pytest.raises(ValueError) as exc_info:
            palletjack.utils.Chunking._chunk_dataframe(df, 2)

        assert "Dataframe chunk is only one row (index 0), further chunking impossible" in str(exc_info.value)

    def test_recursive_dataframe_chunking_recurses_on_first_chunk_too_large(self, mocker):
        #: 5 rows, 2 chunks, first chunk gets 4 instead of three.
        #: two total calls: once for 5 elements, once for 4. first call chunks to 4 and 1, next chunks 4 into 3 and 1
        df = pd.DataFrame(["a", "b", "c", "d", "e"], columns=["foo"])
        mocker.patch("palletjack.utils.pd.DataFrame.spatial.to_featureset", return_value=mocker.Mock())
        mocker.patch("palletjack.utils.Chunking._ceildiv")
        sys_mock = mocker.patch("palletjack.utils.sys.getsizeof", side_effect=["foo", 4, "foo", 2, 2, 1])
        mocker.patch(
            "palletjack.utils.Chunking._chunk_dataframe",
            side_effect=[
                [df.iloc[:4], df.iloc[4:]],  #: first chunking gives 4 and 1
                [df.iloc[:3], df.iloc[3:4]],  #: second chunking gives 3 and 1 (3 being the max size)
            ],
        )
        df_list = palletjack.utils.Chunking._recursive_dataframe_chunking(df, 3)

        tm.assert_frame_equal(df_list[0], df.iloc[:3])
        tm.assert_frame_equal(df_list[1], df.iloc[3:4])
        tm.assert_frame_equal(df_list[2], df.iloc[4:])
        assert sys_mock.call_count == 6

    def test_recursive_dataframe_chunking_recurses_on_second_chunk_too_large(self, mocker):
        #: 5 rows, 2 chunks, first chunk gets 1 instead of three, second gets 4
        #: two total calls: once for 5 elements, once for 4. first call chunks to 1 and 4, next chunks 4 into 3 and 1
        df = pd.DataFrame(["a", "b", "c", "d", "e"], columns=["foo"])
        mocker.patch("palletjack.utils.pd.DataFrame.spatial.to_featureset", return_value=mocker.Mock())
        mocker.patch("palletjack.utils.Chunking._ceildiv")
        sys_mock = mocker.patch("palletjack.utils.sys.getsizeof", side_effect=["foo", 1, 4, "foo", 2, 2])
        mocker.patch(
            "palletjack.utils.Chunking._chunk_dataframe",
            side_effect=[
                [df.iloc[:1], df.iloc[1:]],  #: first chunking gives 1 and 4
                [df.iloc[1:4], df.iloc[4:]],  #: second chunking gives 3 and 1 (3 being the max size)
            ],
        )

        df_list = palletjack.utils.Chunking._recursive_dataframe_chunking(df, 3)

        tm.assert_frame_equal(df_list[0], df.iloc[:1])
        tm.assert_frame_equal(df_list[1], df.iloc[1:4])
        tm.assert_frame_equal(df_list[2], df.iloc[4:])
        assert sys_mock.call_count == 6

    def test_recursive_dataframe_chunking_recurses_on_middle_chunk_too_large(self, mocker):
        #: 5 rows, 3 chunks, first chunk gets 1, second gets 3, third gets 1
        #: two total calls: once for 5 elements, once for 3.
        df = pd.DataFrame(["a", "b", "c", "d", "e"], columns=["foo"])
        mocker.patch("palletjack.utils.pd.DataFrame.spatial.to_featureset", return_value=mocker.Mock())
        mocker.patch("palletjack.utils.Chunking._ceildiv")
        sys_mock = mocker.patch("palletjack.utils.sys.getsizeof", side_effect=["foo", 1, 3, "foo", 2, 1, 1])
        mocker.patch(
            "palletjack.utils.Chunking._chunk_dataframe",
            side_effect=[
                [df.iloc[:1], df.iloc[1:4], df.iloc[4:]],  #: first chunking gives 1, 3, 1
                [df.iloc[1:3], df.iloc[3:4]],  #: second chunking gives 2 and 1 (2 being the max size)
            ],
        )

        df_list = palletjack.utils.Chunking._recursive_dataframe_chunking(df, 2)

        tm.assert_frame_equal(df_list[0], df.iloc[:1])
        tm.assert_frame_equal(df_list[1], df.iloc[1:3])
        tm.assert_frame_equal(df_list[2], df.iloc[3:4])
        tm.assert_frame_equal(df_list[3], df.iloc[4:])
        assert sys_mock.call_count == 7

    def test_recursive_dataframe_chunking_doesnt_recurse_when_not_needed(self, mocker):
        #: 5 rows, 2 chunks, first chunk gets 3, second gets 2
        #: only one call: first breaks into 3 and 2
        df = pd.DataFrame(["a", "b", "c", "d", "e"], columns=["foo"])
        mocker.patch("palletjack.utils.pd.DataFrame.spatial.to_featureset", return_value=mocker.Mock())
        mocker.patch("palletjack.utils.Chunking._ceildiv")
        sys_mock = mocker.patch("palletjack.utils.sys.getsizeof", side_effect=["foo", 3, 2])
        mocker.patch(
            "palletjack.utils.Chunking._chunk_dataframe",
            side_effect=[
                [df.iloc[:3], df.iloc[3:]],  #: first chunking gives 3, 2
            ],
        )

        df_list = palletjack.utils.Chunking._recursive_dataframe_chunking(df, 3)

        tm.assert_frame_equal(df_list[0], df.iloc[:3])
        tm.assert_frame_equal(df_list[1], df.iloc[3:])
        assert sys_mock.call_count == 3

    def test_recursive_dataframe_chunking_doesnt_chunk_when_not_needed(self, mocker):
        #: 5 rows, 2 chunks, first chunk gets 3, second gets 2
        #: only one call: first breaks into 3 and 2
        df = pd.DataFrame(["a", "b", "c", "d", "e"], columns=["foo"])
        mocker.patch("palletjack.utils.pd.DataFrame.spatial.to_featureset", return_value=mocker.Mock())
        mocker.patch("palletjack.utils.Chunking._ceildiv")
        sys_mock = mocker.patch("palletjack.utils.sys.getsizeof", side_effect=["foo", 5])
        chunking_mock = mocker.patch(
            "palletjack.utils.Chunking._chunk_dataframe",
            side_effect=[
                [df],  #: no chunk needed
            ],
        )

        df_list = palletjack.utils.Chunking._recursive_dataframe_chunking(df, 5)

        tm.assert_frame_equal(df_list[0], df)
        chunking_mock.assert_called_once()
        assert sys_mock.call_count == 2

    def test_recursive_dataframe_chunking_raises_when_single_row_is_too_big(self, mocker):
        #: First chunk triggers a recursive call, which returns a single row and should error out

        max_bytes = 4
        error_text = "Dataframe chunk is only one row (index 2), further chunking impossible"

        df = pd.DataFrame(["a", "b", "c", "d", "e"], columns=["foo"])
        mocker.patch("palletjack.utils.pd.DataFrame.spatial.to_featureset", return_value=mocker.Mock())
        mocker.patch("palletjack.utils.Chunking._ceildiv")
        sys_mock = mocker.patch("palletjack.utils.sys.getsizeof", side_effect=["foo", 5, "foo", 5])
        chunking_mock = mocker.patch(
            "palletjack.utils.Chunking._chunk_dataframe",
            side_effect=[
                [df.iloc[:3], df.iloc[3:]],
                ValueError(error_text),
            ],
        )

        with pytest.raises(ValueError) as exc_info:
            df_list = palletjack.utils.Chunking._recursive_dataframe_chunking(df, max_bytes)

        assert error_text in str(exc_info.value)

    def test_build_upload_json_calls_null_string_fixer_appropriate_number_of_times(self, mocker):
        mock_df = mocker.Mock()
        mock_string_fixer = mocker.patch("palletjack.utils.fix_numeric_empty_strings")
        mock_string_fixer.return_value.to_geojson = "new_json"
        mocker.patch("palletjack.utils.sys.getsizeof")
        mocker.patch("palletjack.utils.Chunking._recursive_dataframe_chunking", return_value=[mocker.Mock()] * 5)

        json_list = palletjack.utils.Chunking.build_upload_json(mock_df, "foo_dict")

        assert len(json_list) == 5
        assert json_list == ["new_json"] * 5
        assert mock_string_fixer.call_count == 5


class TestChunker:
    def test_chunker(self):
        sequence = ["a", "b", "c", "d", "e", "f", "g"]
        chunks = [chunk for chunk in palletjack.utils.chunker(sequence, 3)]

        assert chunks == [["a", "b", "c"], ["d", "e", "f"], ["g"]]


class TestSEDFtoGDF:
    def test_sedf_to_gdf_uses_wkid_when_missing_latestwkid(self, mocker):
        gdf_mock = mocker.patch("palletjack.utils.gpd.GeoDataFrame").return_value

        df_mock = mocker.Mock()
        df_mock.spatial.sr = mocker.Mock(spec=palletjack.utils.arcgis.geometry.SpatialReference)
        df_mock.spatial.sr.wkid = "foo"

        palletjack.utils.sedf_to_gdf(df_mock)

        gdf_mock.set_crs.assert_called_with("foo", inplace=True)

    def test_sedf_to_gdf_uses_sedf_geometry_column(self, mocker):
        mock_sedf = mocker.Mock(**{"spatial.name": "FOOSHAPE"})  # , "spatial.sr": {"wkid": "foo"}})
        mock_sedf.spatial.sr = mocker.Mock(spec=palletjack.utils.arcgis.geometry.SpatialReference)
        mock_sedf.spatial.sr.wkid = "foo"
        gpd_mock = mocker.patch("palletjack.utils.gpd.GeoDataFrame")

        palletjack.utils.sedf_to_gdf(mock_sedf)

        gpd_mock.assert_called_with(mock_sedf, geometry="FOOSHAPE")


class TestConvertToGDF:
    #: Tests created by copilot, checked and fixed by hand

    def test_convert_to_gdf_returns_gdf_if_already_gdf(self, mocker):
        gdf_mock = mocker.Mock(spec=palletjack.utils.gpd.GeoDataFrame)
        result = palletjack.utils.convert_to_gdf(gdf_mock)
        assert result is gdf_mock

    def test_convert_to_gdf_returns_gdf_with_none_geometry_for_regular_df(self, mocker):
        df_mock = mocker.Mock()
        #: Simulate KeyError when accessing .spatial.geometry_type
        type(df_mock).spatial = property(lambda self: (_ for _ in ()).throw(KeyError()))
        gdf_mock = mocker.patch("palletjack.utils.gpd.GeoDataFrame")
        #: since we're mocking gpd.GeoDataFrame, the isinstance check errors, so just make it return false
        mocker.patch("palletjack.utils.isinstance", return_value=False)

        result = palletjack.utils.convert_to_gdf(df_mock)

        gdf_mock.assert_called_with(df_mock, geometry=None)
        assert result == gdf_mock.return_value

    def test_convert_to_gdf_handles_spatially_enabled_dataframe(self, mocker):
        df_mock = mocker.Mock()
        df_mock.spatial.geometry_type = "Point"
        df_mock.spatial.name = "SHAPE"
        df_mock.spatial.sr.latestWkid = 4326
        gdf_mock = mocker.patch("palletjack.utils.gpd.GeoDataFrame").return_value
        gdf_mock.set_crs = mocker.Mock()
        #: since we're mocking gpd.GeoDataFrame, the isinstance check errors, so just make it return false
        mocker.patch("palletjack.utils.isinstance", return_value=False)

        result = palletjack.utils.convert_to_gdf(df_mock)

        gdf_mock.set_crs.assert_called_with(4326, inplace=True)
        assert result == gdf_mock

    def test_convert_to_gdf_handles_spatially_enabled_dataframe_uses_wkid_instead_of_lakestwkid(self, mocker):
        df_mock = mocker.Mock()
        df_mock.spatial.geometry_type = "Point"
        df_mock.spatial.name = "SHAPE"

        #: spec out sr so that we get an AttributeError if we try to access latestWkid
        sr_mock = mocker.Mock(spec="palletjack.utils.arcgis.geometry.SpatialReference")
        sr_mock.wkid = 4326
        df_mock.spatial.sr = sr_mock

        gdf_mock = mocker.patch("palletjack.utils.gpd.GeoDataFrame").return_value
        gdf_mock.set_crs = mocker.Mock()
        #: since we're mocking gpd.GeoDataFrame, the isinstance check errors, so just make it return false
        mocker.patch("palletjack.utils.isinstance", return_value=False)

        result = palletjack.utils.convert_to_gdf(df_mock)

        gdf_mock.set_crs.assert_called_with(4326, inplace=True)
        assert result == gdf_mock
