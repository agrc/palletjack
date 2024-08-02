import logging
import re

import numpy as np
import palletjack
import pandas as pd
import pytest
from pandas import testing as tm


class TestAPIGeocoder:
    def test_geocode_dataframe_calls_with_right_args(self, mocker):
        utils_mock = mocker.patch("palletjack.transform.utils", autospec=True)
        utils_mock.Geocoding.validate_api_key.return_value = "valid"
        utils_mock.rename_columns_for_agol.return_value = {}
        utils_mock.Geocoding.geocode_addr.side_effect = [
            (123, 456, 100.0, "foo_addr"),
            (789, 101, 100.0, "bar_addr"),
        ]
        mocker.patch("palletjack.transform.pd.DataFrame.spatial")

        geocoder = palletjack.transform.APIGeocoder("foo")

        test_df = pd.DataFrame(
            {
                "street": ["4315 S 2700 W", "4501 S Constitution Blvd"],
                "zip": ["84129", "84129"],
            }
        )

        geocoder.geocode_dataframe(test_df, "street", "zip", 3857)

        assert utils_mock.Geocoding.geocode_addr.call_args.args == (
            "4501 S Constitution Blvd",
            "84129",
            "foo",
            (0.015, 0.03),
        )
        assert utils_mock.Geocoding.geocode_addr.call_args.kwargs == {"spatialReference": "3857"}

    def test_geocode_dataframe_handles_street_zone_fields_with_invalid_python_names(self, mocker):
        utils_mock = mocker.patch("palletjack.transform.utils", autospec=True)
        utils_mock.Geocoding.validate_api_key.return_value = "valid"
        utils_mock.rename_columns_for_agol.return_value = {}
        utils_mock.Geocoding.geocode_addr.side_effect = [
            (123, 456, 100.0, "foo_addr"),
            (789, 101, 100.0, "bar_addr"),
        ]
        mocker.patch("palletjack.transform.pd.DataFrame.spatial")

        geocoder = palletjack.transform.APIGeocoder("foo")

        test_df = pd.DataFrame(
            {
                "Physical Address Street": ["4315 S 2700 W", "4501 S Constitution Blvd"],
                "Physical Address Zip Code": ["84129", "84129"],
            }
        )

        geocoder.geocode_dataframe(test_df, "Physical Address Street", "Physical Address Zip Code", 3857)

        assert utils_mock.Geocoding.geocode_addr.call_args.args == (
            "4501 S Constitution Blvd",
            "84129",
            "foo",
            (0.015, 0.03),
        )
        assert utils_mock.Geocoding.geocode_addr.call_args.kwargs == {"spatialReference": "3857"}

    def test_geocode_dataframe_passes_kwargs_through_to_util_method(self, mocker):
        utils_mock = mocker.patch("palletjack.transform.utils", autospec=True)
        utils_mock.Geocoding.validate_api_key.return_value = "valid"
        utils_mock.rename_columns_for_agol.return_value = {}
        utils_mock.Geocoding.geocode_addr.side_effect = [
            (123, 456, 100.0, "foo_addr"),
            (789, 101, 100.0, "bar_addr"),
        ]
        mocker.patch("palletjack.transform.pd.DataFrame.spatial")

        geocoder = palletjack.transform.APIGeocoder("foo")

        test_df = pd.DataFrame(
            {
                "street": ["4315 S 2700 W", "4501 S Constitution Blvd"],
                "zip": ["84129", "84129"],
            }
        )

        geocoder.geocode_dataframe(test_df, "street", "zip", 3857, acceptScore=80)

        assert utils_mock.Geocoding.geocode_addr.call_args.args == (
            "4501 S Constitution Blvd",
            "84129",
            "foo",
            (0.015, 0.03),
        )
        assert utils_mock.Geocoding.geocode_addr.call_args.kwargs == {"spatialReference": "3857", "acceptScore": 80}

    def test_geocode_dataframe_builds_output_dataframe(self, mocker, caplog):
        caplog.set_level(logging.DEBUG)
        requests_mock = mocker.patch("palletjack.utils.requests", autospec=True)
        mocker.patch("palletjack.utils.sleep")
        mocker.patch("palletjack.utils.Geocoding.validate_api_key", return_value="valid")

        first_response = mocker.Mock()
        first_response.json.return_value = {
            "status": 200,
            "result": {"location": {"x": 123, "y": 456}, "score": 100.0, "matchAddress": "bar"},
        }
        first_response.status_code = 200

        second_response = mocker.Mock()
        second_response.json.return_value = {
            "status": 200,
            "result": {"location": {"x": 789, "y": 101}, "score": 100.0, "matchAddress": "baz"},
        }
        second_response.status_code = 200

        requests_mock.get.side_effect = [first_response, second_response]

        mocker.patch("palletjack.transform.pd.DataFrame.spatial.from_xy", side_effect=lambda df, x, y, sr: df)

        geocoder = palletjack.transform.APIGeocoder("foo")

        input_df = pd.DataFrame(
            {
                "street": ["4315 S 2700 W", "4501 S Constitution Blvd"],
                "zip": ["84129", "84129"],
            }
        )

        geocoded_df = geocoder.geocode_dataframe(input_df, "street", "zip", 26912)

        test_df = pd.DataFrame(
            {
                "street": ["4315 S 2700 W", "4501 S Constitution Blvd"],
                "zip": ["84129", "84129"],
                "x": [123, 789],
                "y": [456, 101],
                "score": [100.0, 100.0],
                "matchAddress": ["bar", "baz"],
            }
        )

        tm.assert_frame_equal(test_df, geocoded_df)

    def test_geocode_dataframe_continues_after_failed_row(self, mocker, caplog):
        caplog.set_level(logging.DEBUG)
        requests_mock = mocker.patch("palletjack.utils.requests", autospec=True)
        mocker.patch("palletjack.utils.sleep")
        mocker.patch("palletjack.utils.Geocoding.validate_api_key", return_value="valid")

        bad_response = mocker.Mock()
        bad_response.status_code = 500

        good_response = mocker.Mock()
        good_response.json.return_value = {
            "status": 200,
            "result": {"location": {"x": 123, "y": 456}, "score": 100.0, "matchAddress": "bar"},
        }
        good_response.status_code = 200

        requests_mock.get.side_effect = [bad_response, bad_response, bad_response, bad_response, good_response]

        mocker.patch("palletjack.transform.pd.DataFrame.spatial.from_xy", side_effect=lambda df, x, y, sr: df)

        geocoder = palletjack.transform.APIGeocoder("foo")

        input_df = pd.DataFrame(
            {
                "street": ["4315 S 2700 W", "4501 S Constitution Blvd"],
                "zip": ["84129", "84129"],
            }
        )

        geocoded_df = geocoder.geocode_dataframe(input_df, "street", "zip", 26912)

        test_df = pd.DataFrame(
            {
                "street": ["4315 S 2700 W", "4501 S Constitution Blvd"],
                "zip": ["84129", "84129"],
                "x": [0, 123],
                "y": [0, 456],
                "score": [0.0, 100.0],
                "matchAddress": ["No API response", "bar"],
            }
        )

        tm.assert_frame_equal(test_df, geocoded_df)
        assert "Did not receive a valid geocoding response; status code: 500" in caplog.messages
        assert palletjack.utils.requests.get.call_count == 5
        assert palletjack.utils.sleep.call_count == 5

    def test_geocode_dataframe_warns_on_empty_input(self, mocker):
        utils_mock = mocker.patch("palletjack.transform.utils", autospec=True)
        utils_mock.Geocoding.validate_api_key.return_value = "valid"
        mocker.patch("palletjack.transform.pd.DataFrame.spatial.from_xy")

        geocoder = palletjack.transform.APIGeocoder("foo")

        test_df = pd.DataFrame(columns=["street", "zip"])

        with pytest.warns(RuntimeWarning) as record:
            geocoder.geocode_dataframe(test_df, "street", "zip", 3857)
        assert record[0].message.args[0] == "No records to geocode (empty dataframe)"

    def test_geocode_dataframe_warns_on_empty_output(self, mocker):
        utils_mock = mocker.patch("palletjack.transform.utils", autospec=True)
        utils_mock.Geocoding.validate_api_key.return_value = "valid"
        utils_mock.rename_columns_for_agol.return_value = {}
        utils_mock.Geocoding.geocode_addr.side_effect = [
            (123, 456, 100.0, "foo_addr"),
            (789, 101, 100.0, "bar_addr"),
        ]
        mocker.patch(
            "palletjack.transform.pd.DataFrame.spatial.from_xy",
            return_value=pd.DataFrame(columns=["street", "zip", "x", "y", "score", "matchAddress"]),
        )

        geocoder = palletjack.transform.APIGeocoder("foo")

        test_df = pd.DataFrame(
            {
                "street": ["4315 S 2700 W", "4501 S Constitution Blvd"],
                "zip": ["84129", "84129"],
            }
        )

        with pytest.warns(RuntimeWarning) as record:
            geocoder.geocode_dataframe(test_df, "street", "zip", 3857)
        assert record[0].message.args[0] == "Empty spatial dataframe after geocoding"

    def test_geocode_dataframe_raises_on_invalid_key(self, mocker):
        req_mock = mocker.patch("palletjack.utils.requests", autospec=True)
        response_mock = mocker.Mock()
        response_mock.json.return_value = {"status": 400, "message": "Invalid API key"}
        req_mock.get.return_value = response_mock

        with pytest.raises(ValueError):
            palletjack.transform.APIGeocoder("foo")

    # : TODO: test api validate responses in __init__


class TestFeatureServiceMerging:
    def test_update_live_data_with_new_data_works_normally(self):
        new_dataframe = pd.DataFrame(
            {
                "col1": [10, 20, 30],
                "col2": [40, 50, 60],
                "key": ["a", "b", "c"],
            }
        )
        live_dataframe = pd.DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": [4, 5, 6],
                "key": ["a", "b", "c"],
            }
        )

        joined = palletjack.transform.FeatureServiceMerging.update_live_data_with_new_data(
            live_dataframe, new_dataframe, "key"
        )

        expected = pd.DataFrame(
            {
                "key": ["a", "b", "c"],
                "col1": [10, 20, 30],
                "col2": [40, 50, 60],
            }
        ).convert_dtypes()

        pd.testing.assert_frame_equal(joined, expected, check_like=True)

    def test_update_live_data_with_new_data_only_updates_common_rows(self):
        new_dataframe = pd.DataFrame(
            {
                "col1": [20, 30],
                "col2": [50, 60],
                "key": ["b", "c"],
            }
        )
        live_dataframe = pd.DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": [4, 5, 6],
                "key": ["a", "b", "c"],
            }
        )

        joined = palletjack.transform.FeatureServiceMerging.update_live_data_with_new_data(
            live_dataframe, new_dataframe, "key"
        )

        expected = pd.DataFrame(
            {
                "col1": [1, 20, 30],
                "col2": [4, 50, 60],
                "key": ["a", "b", "c"],
            }
        ).convert_dtypes()

        pd.testing.assert_frame_equal(joined, expected, check_like=True)

    def test_update_live_data_with_new_data_warns_on_missing_keys_and_handles_ints(self):
        new_dataframe = pd.DataFrame(
            {
                "col1": [10, 20, 30, 80],
                "col2": [40, 50, 60, 70],
                "key": ["a", "b", "c", "d"],
            }
        )
        live_dataframe = pd.DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": [4, 5, 6],
                "key": ["a", "b", "c"],
            }
        )

        with pytest.warns(RuntimeWarning) as warning:
            joined = palletjack.transform.FeatureServiceMerging.update_live_data_with_new_data(
                live_dataframe, new_dataframe, "key"
            )

        expected = pd.DataFrame(
            {
                "key": ["a", "b", "c"],
                "col1": [10, 20, 30],
                "col2": [40, 50, 60],
            }
        ).convert_dtypes()

        pd.testing.assert_frame_equal(joined, expected, check_like=True)
        assert (
            "The following keys from the new data were not found in the existing dataset: ['d']"
            in warning[0].message.args[0]
        )

    def test_get_live_dataframe_calls_each_method(self, mocker):
        mock_arcgis = mocker.patch("palletjack.transform.arcgis")
        mock_gis = mocker.Mock()

        palletjack.transform.FeatureServiceMerging.get_live_dataframe(mock_gis, "itemid", 42)

        mock_gis.content.get.assert_called_once_with("itemid")
        mock_arcgis.features.FeatureLayer.fromitem.assert_called_once_with(
            mock_gis.content.get.return_value, layer_id=42
        )
        mock_arcgis.features.FeatureLayer.fromitem.return_value.query.assert_called_once_with(as_df=True)

    def test_get_live_dataframe_raises_error_on_get_error(self, mocker):
        mock_arcgis = mocker.patch("palletjack.transform.arcgis")
        mock_gis = mocker.Mock()
        mock_gis.content.get.side_effect = ValueError("get error")

        with pytest.raises(RuntimeError) as error:
            palletjack.transform.FeatureServiceMerging.get_live_dataframe(mock_gis, "itemid")

        assert "Failed to load live dataframe" in str(error.value)
        assert isinstance(error.value.__cause__, ValueError)
        assert error.value.__cause__.args[0] == "get error"

    def test_get_live_dataframe_raises_error_on_fromitem_error(self, mocker):
        mock_arcgis = mocker.patch("palletjack.transform.arcgis")
        mock_gis = mocker.Mock()
        mock_arcgis.features.FeatureLayer.fromitem.side_effect = ValueError("fromitem error")

        with pytest.raises(RuntimeError) as error:
            palletjack.transform.FeatureServiceMerging.get_live_dataframe(mock_gis, "itemid")

        assert "Failed to load live dataframe" in str(error.value)
        assert isinstance(error.value.__cause__, ValueError)
        assert error.value.__cause__.args[0] == "fromitem error"

    def test_get_live_dataframe_raises_error_on_query_error(self, mocker):
        mock_arcgis = mocker.patch("palletjack.transform.arcgis")
        mock_gis = mocker.Mock()
        mock_arcgis.features.FeatureLayer.fromitem.return_value.query.side_effect = ValueError("query error")

        with pytest.raises(RuntimeError) as error:
            palletjack.transform.FeatureServiceMerging.get_live_dataframe(mock_gis, "itemid")

        assert "Failed to load live dataframe" in str(error.value)
        assert isinstance(error.value.__cause__, ValueError)
        assert error.value.__cause__.args[0] == "query error"


class TestNullableIntFixing:
    def test_switch_to_nullable_int_casts_float_field_with_nan(self):
        df = pd.DataFrame(
            {
                "a": [1, 2, np.nan],
            }
        )

        retyped_df = palletjack.transform.DataCleaning.switch_to_nullable_int(df, ["a"])

        test_df = pd.DataFrame([1, 2, pd.NA], columns=["a"], dtype="Int64")

        tm.assert_frame_equal(retyped_df, test_df)

    def test_switch_to_nullable_int_doesnt_change_other_float_field(self):
        df = pd.DataFrame(
            {
                "a": [1, 2, np.nan],
                "b": [1.1, 1.2, 1.3],
            }
        )

        retyped_df = palletjack.transform.DataCleaning.switch_to_nullable_int(df, ["a"])

        test_df = pd.DataFrame(
            {"a": pd.Series([1, 2, pd.NA], dtype="Int64"), "b": pd.Series([1.1, 1.2, 1.3], dtype="float64")}
        )

        tm.assert_frame_equal(retyped_df, test_df)

    def test_switch_to_nullable_int_raises_on_true_float_field(self):
        df = pd.DataFrame(
            {
                "a": [1, 2, np.nan],
                "b": [1.1, 1.2, 1.3],
            }
        )

        with pytest.raises(
            TypeError,
            match=re.escape("Cannot convert one or more fields to nullable ints. Check for non-int/non-np.nan values."),
        ):
            retyped_df = palletjack.transform.DataCleaning.switch_to_nullable_int(df, ["a", "b"])

    def test_switch_to_nullable_int_comma_thousands_separator(self):
        df = pd.DataFrame(
            {
                "a": ["1", "2", "3,000"],
            }
        )

        retyped_df = palletjack.transform.DataCleaning.switch_to_nullable_int(df, ["a"])

        test_df = pd.DataFrame([1, 2, 3000], columns=["a"], dtype="Int64")

        tm.assert_frame_equal(retyped_df, test_df)

    def test_switch_to_nullable_int_comma_thousands_separator_mixed_input_types(self):
        df = pd.DataFrame(
            {
                "a": [1, "2", "3,000"],
            }
        )

        retyped_df = palletjack.transform.DataCleaning.switch_to_nullable_int(df, ["a"])

        test_df = pd.DataFrame([1, 2, 3000], columns=["a"], dtype="Int64")

        tm.assert_frame_equal(retyped_df, test_df)

    def test_switch_to_nullable_int_casts_string_field_with_empty_string(self):
        df = pd.DataFrame(
            {
                "a": ["1", "2", ""],
            }
        )

        retyped_df = palletjack.transform.DataCleaning.switch_to_nullable_int(df, ["a"])

        test_df = pd.DataFrame([1, 2, pd.NA], columns=["a"], dtype="Int64")

        tm.assert_frame_equal(retyped_df, test_df)


class TestFloatFixing:
    def test_switch_to_float_all_string_values(self):
        df = pd.DataFrame(
            {
                "a": ["1", "2", "3"],
            }
        )

        retyped_df = palletjack.transform.DataCleaning.switch_to_float(df, ["a"])

        test_df = pd.DataFrame([1.0, 2.0, 3.0], columns=["a"], dtype="float")

        tm.assert_frame_equal(retyped_df, test_df)

    def test_switch_to_float_string_values_with_nan(self):
        df = pd.DataFrame(
            {
                "a": ["1", "2", np.nan],
            }
        )

        retyped_df = palletjack.transform.DataCleaning.switch_to_float(df, ["a"])

        test_df = pd.DataFrame([1.0, 2.0, np.nan], columns=["a"], dtype="float")

        tm.assert_frame_equal(retyped_df, test_df)

    def test_switch_to_float_string_values_with_empty_string(self):
        df = pd.DataFrame(
            {
                "a": ["1", "2", ""],
            }
        )

        retyped_df = palletjack.transform.DataCleaning.switch_to_float(df, ["a"])

        test_df = pd.DataFrame([1.0, 2.0, np.nan], columns=["a"], dtype="float")

        tm.assert_frame_equal(retyped_df, test_df)

    def test_switch_to_float_comma_thousands_separator(self):
        df = pd.DataFrame(
            {
                "a": ["1", "2", "3,000"],
            }
        )

        retyped_df = palletjack.transform.DataCleaning.switch_to_float(df, ["a"])

        test_df = pd.DataFrame([1.0, 2.0, 3000.0], columns=["a"], dtype="float")

        tm.assert_frame_equal(retyped_df, test_df)

    def test_switch_to_float_comma_thousands_separator_mixed_input_types(self):
        df = pd.DataFrame(
            {
                "a": [1, 2.0, "3,000"],
            }
        )

        retyped_df = palletjack.transform.DataCleaning.switch_to_float(df, ["a"])

        test_df = pd.DataFrame([1.0, 2.0, 3000.0], columns=["a"], dtype="float")

        tm.assert_frame_equal(retyped_df, test_df)

    def test_switch_to_float_raises_on_uncastable_text(self):
        df = pd.DataFrame(
            {
                "a": [1, 2, np.nan],
                "b": [1.1, 1.2, "foo"],
            }
        )

        with pytest.raises(
            TypeError,
            match=re.escape("Cannot convert one or more fields to floats. Check for non-float/non-null values."),
        ):
            retyped_df = palletjack.transform.DataCleaning.switch_to_float(df, ["a", "b"])


class TestSwitchSeriesToNumericDtype:
    def test_switch_series_to_numeric_dtype_ints_to_Int64(self):
        series1 = pd.Series([1, 2, 3, 4, 5])
        test_series = pd.Series([1, 2, 3, 4, 5], dtype="Int64")
        assert palletjack.transform.DataCleaning._switch_series_to_numeric_dtype(series1, "Int64").equals(test_series)

    def test_switch_series_to_numeric_dtype_str_to_float_with_thousands(self):
        series2 = pd.Series(["1,000", "2,000", "3,000", "4,000", "5,000"])
        test_series = pd.Series([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])
        assert palletjack.transform.DataCleaning._switch_series_to_numeric_dtype(series2, "float").equals(test_series)

    def test_switch_series_to_numeric_dtype_mixed_strings_ints_to_Int64(self):
        series3 = pd.Series(["1,000", 2, "3,000", 4, "5,000"])
        test_series = pd.Series([1000, 2, 3000, 4, 5000], dtype="Int64")
        assert palletjack.transform.DataCleaning._switch_series_to_numeric_dtype(series3, "Int64").equals(test_series)

    def test_switch_series_to_numeric_dtype_mixed_strings_ints_to_float(self):
        series3 = pd.Series(["1,000", 2, "3,000", 4, "5,000"])
        test_series = pd.Series([1000.0, 2.0, 3000.0, 4.0, 5000.0])
        assert palletjack.transform.DataCleaning._switch_series_to_numeric_dtype(series3, "float").equals(test_series)

    def test_switch_series_to_numeric_dtype_ints_with_nan_to_Int64(self):
        series4 = pd.Series([1, 2, np.nan])
        test_series = pd.Series([1, 2, pd.NA], dtype="Int64")
        assert palletjack.transform.DataCleaning._switch_series_to_numeric_dtype(series4, "Int64").equals(test_series)

    def test_switch_series_to_numeric_dtype_raises_on_non_numeric(self):
        series5 = pd.Series(["a", "b", "c", "d", "e"])
        with pytest.raises(ValueError):
            palletjack.transform.DataCleaning._switch_series_to_numeric_dtype(series5, "float")


class TestDatetimeSwitching:
    def test_switch_to_datetime_handles_multiple_fields(self):
        df = pd.DataFrame(
            {
                "a": ["2020-04-29T01:09:29+00:00", "2020-04-29T01:09:29+00:00"],
                "b": ["2015-09-02T23:08:12+00:00", "2015-09-02T23:08:12+00:00"],
            }
        )

        retyped_df = palletjack.transform.DataCleaning.switch_to_datetime(df, ["a", "b"])

        test_df = pd.DataFrame(
            {"a": [pd.Timestamp("2020-04-29 01:09:29")] * 2, "b": [pd.Timestamp("2015-09-02 23:08:12")] * 2}
        )

        tm.assert_frame_equal(retyped_df, test_df)
        assert retyped_df["a"].dtype == np.dtype("datetime64[ns]")
        assert retyped_df["b"].dtype == np.dtype("datetime64[ns]")

    def test_switch_to_datetime_handles_both_empty_string_and_None(self):
        df = pd.DataFrame(
            {
                "a": ["", "2020-04-29T01:09:29+00:00"],
                "b": [None, "2015-09-02T23:08:12+00:00"],
            }
        )

        retyped_df = palletjack.transform.DataCleaning.switch_to_datetime(df, ["a", "b"])

        test_df = pd.DataFrame(
            {
                "a": [pd.Timestamp(None), pd.Timestamp("2020-04-29 01:09:29")],
                "b": [pd.Timestamp(None), pd.Timestamp("2015-09-02 23:08:12")],
            }
        )

        tm.assert_frame_equal(retyped_df, test_df)
        assert retyped_df["a"].dtype == np.dtype("datetime64[ns]")
        assert retyped_df["b"].dtype == np.dtype("datetime64[ns]")

    def test_switch_to_datetime_handles_empty_column(self):
        df = pd.DataFrame(
            {
                "a": [None, None],
                "b": ["2015-09-02T23:08:12+00:00", "2015-09-02T23:08:12+00:00"],
            }
        )

        retyped_df = palletjack.transform.DataCleaning.switch_to_datetime(df, ["a", "b"])

        test_df = pd.DataFrame(
            {
                "a": [pd.Timestamp(None), pd.Timestamp(None)],
                "b": [pd.Timestamp("2015-09-02 23:08:12"), pd.Timestamp("2015-09-02 23:08:12")],
            }
        )

        tm.assert_frame_equal(retyped_df, test_df)
        assert retyped_df["a"].dtype == np.dtype("datetime64[ns]")
        assert retyped_df["b"].dtype == np.dtype("datetime64[ns]")

    def test_switch_to_datetime_handles_units_besides_ns(self, mocker):
        df = pd.DataFrame(
            {
                "b": ["2015-09-02T23:08:12+00:00", "2015-09-02T23:08:12+00:00"],
            }
        )
        df["b"] = pd.to_datetime(df["b"]).dt.as_unit("ms")

        retyped_df = palletjack.transform.DataCleaning.switch_to_datetime(df, ["b"])

        test_df = pd.DataFrame({"b": [pd.Timestamp("2015-09-02 23:08:12"), pd.Timestamp("2015-09-02 23:08:12")]})

        tm.assert_frame_equal(retyped_df, test_df)
        assert retyped_df["b"].dtype == np.dtype("datetime64[ns]")


class TestDataFrameColumnRenaming:
    def test_rename_dataframe_columns_for_agol(self, mocker):
        df = pd.DataFrame(
            {
                "street": ["4315 S 2700 W"],
                "zip": ["84129"],
                "col_with_underscores": ["foo"],
                "Col With Spaces": ["bar"],
                "_starts_with_underscore": ["baz"],
                "1starts_with_number": ["eggs"],
                "1_starts_with_number_and_underscore": ["fried"],
                "includes_1_number": ["ham"],
                "includes!mark": ["brie"],
            }
        )

        renamed_df = palletjack.transform.DataCleaning.rename_dataframe_columns_for_agol(df)

        test_df = pd.DataFrame(
            {
                "street": ["4315 S 2700 W"],
                "zip": ["84129"],
                "col_with_underscores": ["foo"],
                "Col_With_Spaces": ["bar"],
                "starts_with_underscore_": ["baz"],
                "starts_with_number1": ["eggs"],
                "starts_with_number_and_underscore1_": ["fried"],
                "includes_1_number": ["ham"],
                "includes_mark": ["brie"],
            }
        )

        tm.assert_frame_equal(renamed_df, test_df)
