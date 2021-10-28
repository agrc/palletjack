import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from era import data_coupler


class TestFeatureServiceInLineUpdater:

    def test_update_feature_service(self, mocker):
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

        data_coupler.FeatureServiceInLineUpdater.update_feature_service(
            fsupdater_mock, 'foo', ['ZipCode', 'Count', 'Amount', 'Date']
        )

        cursor_mock.updateRow.assert_called_with(['12345', '57', 100.00, '1/1/2022'])
        cursor_mock.updateRow.assert_called_once()


class TestSFTPLoader:

    def test_download_sftp_files_uses_right_credentials(self, mocker):
        sftploader_mock = mocker.Mock()
        sftploader_mock.secrets.KNOWNHOSTS = 'knownhosts_file'
        sftploader_mock.secrets.SFTP_HOST = 'sftp_host'
        sftploader_mock.secrets.SFTP_USERNAME = 'username'
        sftploader_mock.secrets.SFTP_PASSWORD = 'password'
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

        data_coupler.SFTPLoader.download_sftp_files(sftploader_mock)

        context_manager_mock.assert_called_with(
            'sftp_host', username='username', password='password', cnopts='knownhosts_file'
        )

    def test_read_csv_into_dataframe_with_column_names(self, mocker):
        pd_mock = mocker.Mock()
        pd_mock.return_value = pd.DataFrame()
        mocker.patch.object(pd, 'read_csv', new=pd_mock)

        sftploader_mock = mocker.Mock()
        sftploader_mock.download_dir = 'foo'

        column_types = {'bar': np.float64}

        data_coupler.SFTPLoader.read_csv_into_dataframe(sftploader_mock, 'baz', column_types)

        pd_mock.assert_called_with(Path('foo', 'baz'), names=['bar'], dtype=column_types)

    def test_read_csv_into_dataframe_no_column_names(self, mocker):
        pd_mock = mocker.Mock()
        pd_mock.return_value = pd.DataFrame()
        mocker.patch.object(pd, 'read_csv', new=pd_mock)

        sftploader_mock = mocker.Mock()
        sftploader_mock.download_dir = 'foo'

        data_coupler.SFTPLoader.read_csv_into_dataframe(sftploader_mock, 'baz')

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
        reclassifier = data_coupler.ColorRampReclassifier(webmap_item_mock, 'gis')

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
        reclassifier = data_coupler.ColorRampReclassifier(webmap_item_mock, 'gis')

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
        reclassifier = data_coupler.ColorRampReclassifier(webmap_item_mock, 'gis')

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
        reclassifier = data_coupler.ColorRampReclassifier(webmap_item_mock, 'gis')

        with pytest.raises(ValueError) as error_info:
            layer_id = reclassifier._get_layer_id('foo')

        assert 'Could not find "foo" in test map' in str(error_info.value)

    def test_calculate_new_stops_with_manual_numbers(self):
        dataframe = pd.DataFrame({'numbers': [100, 300, 500, 700, 900]})

        stops = data_coupler.ColorRampReclassifier._calculate_new_stops(dataframe, 'numbers', 5)

        assert stops == [100, 279, 458, 637, 816]

    def test_calculate_new_stops_mistmached_column_raises_error(self):
        dataframe = pd.DataFrame({'numbers': [100, 300, 500, 700, 900]})

        with pytest.raises(ValueError) as error_info:
            stops = data_coupler.ColorRampReclassifier._calculate_new_stops(dataframe, 'foo', 5)
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
        reclassifier = data_coupler.ColorRampReclassifier(webmap_item_mock, 'gis')

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
