import logging

import pandas as pd
import pytest
from pandas import testing as tm

import palletjack


class TestAPIGeocoder:

    def test_geocode_dataframe_calls_with_right_args(self, mocker):
        mocker.patch.object(palletjack.transform, 'utils', autospec=True)
        palletjack.transform.utils.rename_columns_for_agol.return_value = {}
        palletjack.transform.utils.geocode_addr.side_effect = [
            (123, 456, 100., 'foo_addr'),
            (789, 101, 100., 'bar_addr'),
        ]
        mocker.patch.object(pd.DataFrame.spatial, 'from_xy')

        geocoder = palletjack.transform.APIGeocoder('foo')

        test_df = pd.DataFrame({
            'street': ['4315 S 2700 W', '4501 S Constitution Blvd'],
            'zip': ['84129', '84129'],
        })

        geocoder.geocode_dataframe(test_df, 'street', 'zip', 3857)

        assert palletjack.transform.utils.geocode_addr.call_args.args == (
            '4501 S Constitution Blvd', '84129', 'foo', (0.015, 0.03)
        )
        assert palletjack.transform.utils.geocode_addr.call_args.kwargs == {'spatialReference': '3857'}

    def test_geocode_dataframe_passes_kwargs_through_to_util_method(self, mocker):
        mocker.patch.object(palletjack.transform, 'utils', autospec=True)
        palletjack.transform.utils.rename_columns_for_agol.return_value = {}
        palletjack.transform.utils.geocode_addr.side_effect = [
            (123, 456, 100., 'foo_addr'),
            (789, 101, 100., 'bar_addr'),
        ]
        mocker.patch.object(pd.DataFrame.spatial, 'from_xy')

        geocoder = palletjack.transform.APIGeocoder('foo')

        test_df = pd.DataFrame({
            'street': ['4315 S 2700 W', '4501 S Constitution Blvd'],
            'zip': ['84129', '84129'],
        })

        geocoder.geocode_dataframe(test_df, 'street', 'zip', 3857, acceptScore=80)

        assert palletjack.transform.utils.geocode_addr.call_args.args == (
            '4501 S Constitution Blvd', '84129', 'foo', (0.015, 0.03)
        )
        assert palletjack.transform.utils.geocode_addr.call_args.kwargs == {
            'spatialReference': '3857',
            'acceptScore': 80
        }

    def test_geocode_dataframe_builds_output_dataframe(self, mocker, caplog):
        caplog.set_level(logging.DEBUG)
        mocker.patch('palletjack.utils.requests', autospec=True)
        mocker.patch('palletjack.utils.sleep')

        first_response = mocker.Mock()
        first_response.json.return_value = {
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
        first_response.status_code = 200

        second_response = mocker.Mock()
        second_response.json.return_value = {
            'status': 200,
            'result': {
                'location': {
                    'x': 789,
                    'y': 101
                },
                'score': 100.,
                'matchAddress': 'baz'
            }
        }
        second_response.status_code = 200

        palletjack.utils.requests.get.side_effect = [first_response, second_response]

        #: Get patch from_xy to just return the dataframe it was passed so we don't have to create a spatial one for
        #: testing
        mocker.patch.object(pd.DataFrame.spatial, 'from_xy')

        def _mock_from_xy(dataframe, x, y, sr=None):
            return dataframe

        pd.DataFrame.spatial.from_xy.side_effect = _mock_from_xy

        geocoder = palletjack.transform.APIGeocoder('foo')

        input_df = pd.DataFrame({
            'street': ['4315 S 2700 W', '4501 S Constitution Blvd'],
            'zip': ['84129', '84129'],
        })

        geocoded_df = geocoder.geocode_dataframe(input_df, 'street', 'zip', 26912)

        test_df = pd.DataFrame({
            'street': ['4315 S 2700 W', '4501 S Constitution Blvd'],
            'zip': ['84129', '84129'],
            'x': [123, 789],
            'y': [456, 101],
            'score': [100., 100.],
            'matchAddress': ['bar', 'baz']
        })

        tm.assert_frame_equal(test_df, geocoded_df)

    def test_geocode_dataframe_continues_after_failed_row(self, mocker, caplog):
        caplog.set_level(logging.DEBUG)
        mocker.patch('palletjack.utils.requests', autospec=True)
        mocker.patch('palletjack.utils.sleep')

        bad_response = mocker.Mock()
        bad_response.status_code = 500

        good_response = mocker.Mock()
        good_response.json.return_value = {
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
        good_response.status_code = 200

        palletjack.utils.requests.get.side_effect = [
            bad_response, bad_response, bad_response, bad_response, good_response
        ]

        #: Get patch from_xy to just return the dataframe it was passed so we don't have to create a spatial one for
        #: testing
        mocker.patch.object(pd.DataFrame.spatial, 'from_xy')

        def _mock_from_xy(dataframe, x, y, sr=None):
            return dataframe

        pd.DataFrame.spatial.from_xy.side_effect = _mock_from_xy

        geocoder = palletjack.transform.APIGeocoder('foo')

        input_df = pd.DataFrame({
            'street': ['4315 S 2700 W', '4501 S Constitution Blvd'],
            'zip': ['84129', '84129'],
        })

        geocoded_df = geocoder.geocode_dataframe(input_df, 'street', 'zip', 26912)

        test_df = pd.DataFrame({
            'street': ['4315 S 2700 W', '4501 S Constitution Blvd'],
            'zip': ['84129', '84129'],
            'x': [0, 123],
            'y': [0, 456],
            'score': [0., 100.],
            'matchAddress': ['No API response', 'bar']
        })

        tm.assert_frame_equal(test_df, geocoded_df)
        assert 'Did not receive a valid geocoding response; status code: 500' in caplog.messages
        assert palletjack.utils.requests.get.call_count == 5
        assert palletjack.utils.sleep.call_count == 5

    def test_geocode_dataframe_warns_on_empty_input(self, mocker):
        mocker.patch.object(palletjack.transform, 'utils', autospec=True)
        # palletjack.transform.utils.geocode_addr.side_effect = [
        #     (123, 456, 100., 'foo_addr'),
        #     (789, 101, 100., 'bar_addr'),
        # ]
        mocker.patch.object(pd.DataFrame.spatial, 'from_xy')

        geocoder = palletjack.transform.APIGeocoder('foo')

        test_df = pd.DataFrame(columns=['street', 'zip'])

        with pytest.warns(RuntimeWarning) as record:
            geocoder.geocode_dataframe(test_df, 'street', 'zip', 3857)
        assert record[0].message.args[0] == 'No records to geocode (empty dataframe)'

    def test_geocode_dataframe_warns_on_empty_output(self, mocker):
        mocker.patch.object(palletjack.transform, 'utils', autospec=True)
        palletjack.transform.utils.rename_columns_for_agol.return_value = {}
        palletjack.transform.utils.geocode_addr.side_effect = [
            (123, 456, 100., 'foo_addr'),
            (789, 101, 100., 'bar_addr'),
        ]
        mocker.patch.object(
            pd.DataFrame.spatial,
            'from_xy',
            return_value=pd.DataFrame(columns=['street', 'zip', 'x', 'y', 'score', 'matchAddress'])
        )

        geocoder = palletjack.transform.APIGeocoder('foo')

        test_df = pd.DataFrame({
            'street': ['4315 S 2700 W', '4501 S Constitution Blvd'],
            'zip': ['84129', '84129'],
        })

        with pytest.warns(RuntimeWarning) as record:
            geocoder.geocode_dataframe(test_df, 'street', 'zip', 3857)
        assert record[0].message.args[0] == 'Empty spatial dataframe after geocoding'

    def test_geocode_dataframe_handles_weird_column_names(self, mocker):
        mocker.patch('palletjack.utils.requests', autospec=True)
        mocker.patch('palletjack.utils.sleep')

        good_response = mocker.Mock()
        good_response.json.return_value = {
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
        good_response.status_code = 200
        palletjack.utils.requests.get.return_value = good_response

        #: Get patch from_xy to just return the dataframe it was passed so we don't have to create a spatial one for
        #: testing
        mocker.patch.object(pd.DataFrame.spatial, 'from_xy')

        def _mock_from_xy(dataframe, x, y, sr=None):
            return dataframe

        pd.DataFrame.spatial.from_xy.side_effect = _mock_from_xy
        geocoder = palletjack.transform.APIGeocoder('foo')
        input_df = pd.DataFrame({
            'street': ['4315 S 2700 W'],
            'zip': ['84129'],
            'col_with_underscores': ['foo'],
            'Col With Spaces': ['bar'],
            '_starts_with_underscore': ['baz'],
            '1starts_with_number': ['eggs'],
            '1_starts_with_number_and_underscore': ['fried'],
            'includes_1_number': ['ham'],
            'includes!mark': ['brie'],
        })

        geocoded_df = geocoder.geocode_dataframe(input_df, 'street', 'zip', 26912)

        out_cols = [
            'street', 'zip', 'col_with_underscores', 'Col_With_Spaces', 'starts_with_underscore_',
            'starts_with_number1', 'starts_with_number_and_underscore1_', 'includes_1_number', 'includes_mark', 'x',
            'y', 'score', 'matchAddress'
        ]

        assert list(geocoded_df.columns) == out_cols
