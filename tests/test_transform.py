import pandas as pd
import pytest

import palletjack


class TestAPIGeocoder:

    def test_geocode_dataframe_calls_with_right_args(self, mocker):
        mocker.patch.object(palletjack.transform, 'utils', autospec=True)
        palletjack.transform.utils.geocode_addr.side_effect = [
            {
                'x': 123,
                'y': 456
            },
            {
                'x': 789,
                'y': 101
            },
        ]
        mocker.patch.object(pd.DataFrame.spatial, 'from_xy')

        geocoder = palletjack.transform.APIGeocoder('foo')

        test_df = pd.DataFrame({
            'street': ['4315 S 2700 W', '4501 S Constitution Blvd'],
            'zip': ['84129', '84129'],
        })

        geocoder.geocode_dataframe(test_df, 'street', 'zip', 3857)

        assert list(palletjack.transform.utils.geocode_addr.call_args.args[0]) == ['4501 S Constitution Blvd', '84129']
        assert palletjack.transform.utils.geocode_addr.call_args.args[1:] == ('street', 'zip', 'foo')
        assert palletjack.transform.utils.geocode_addr.call_args.kwargs == {
            'rate_limits': (0.015, 0.03),
            'spatialReference': '3857'
        }

    def test_geocode_dataframe_passes_kwargs_through_to_util_method(self, mocker):
        mocker.patch.object(palletjack.transform, 'utils', autospec=True)
        palletjack.transform.utils.geocode_addr.side_effect = [
            {
                'x': 123,
                'y': 456
            },
            {
                'x': 789,
                'y': 101
            },
        ]
        mocker.patch.object(pd.DataFrame.spatial, 'from_xy')

        geocoder = palletjack.transform.APIGeocoder('foo')

        test_df = pd.DataFrame({
            'street': ['4315 S 2700 W', '4501 S Constitution Blvd'],
            'zip': ['84129', '84129'],
        })

        geocoder.geocode_dataframe(test_df, 'street', 'zip', 3857, acceptScore=80)

        assert list(palletjack.transform.utils.geocode_addr.call_args.args[0]) == ['4501 S Constitution Blvd', '84129']
        assert palletjack.transform.utils.geocode_addr.call_args.args[1:] == ('street', 'zip', 'foo')
        assert palletjack.transform.utils.geocode_addr.call_args.kwargs == {
            'rate_limits': (0.015, 0.03),
            'spatialReference': '3857',
            'acceptScore': 80
        }
