from pathlib import Path

import numpy as np
import pandas as pd

from era import data_coupler


class TestFeatureServiceInLineUpdater:

    def test_update_feature_service(self, mocker):
        #: We create a mock that will be returned by UpdateCursor's mock's __enter__, thus becoming our context manager.
        #: We then set it's __iter__.return_value to define the data we want it to iterate over.
        contextmanager_mock = mocker.MagicMock()
        contextmanager_mock.__iter__.return_value = [
            ['12345', '42', 123.45, '12/25/2021'],
            ['67890', '18', 67.89, '12/25/2021'],
        ]
        da_mock = mocker.MagicMock()
        da_mock.return_value.__enter__.return_value = contextmanager_mock
        mocker.patch('arcpy.da.UpdateCursor', new=da_mock)

        fsupdater_mock = mocker.Mock()
        fsupdater_mock.data_as_dict = {'12345': {'Count': '57', 'Amount': 100.00, 'Date': '1/1/2022'}}

        data_coupler.FeatureServiceInLineUpdater.update_feature_service(
            fsupdater_mock, 'foo', ['ZipCode', 'Count', 'Amount', 'Date']
        )

        contextmanager_mock.updateRow.assert_called_with(['12345', '57', 100.00, '1/1/2022'])
        contextmanager_mock.updateRow.assert_called_once()


class TestSFTPLoader:

    def test_download_sftp_files_uses_right_credentials(self, mocker):
        sftploader_mock = mocker.Mock()
        sftploader_mock.secrets.KNOWNHOSTS = 'knownhosts_file'
        sftploader_mock.secrets.SFTP_HOST = 'sftp_host'
        sftploader_mock.secrets.SFTP_USERNAME = 'username'
        sftploader_mock.secrets.SFTP_PASSWORD = 'password'
        sftploader_mock.download_dir = 'download_dir'

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
        mocker.patch.object(pd, 'read_csv', new=pd_mock)

        sftploader_mock = mocker.Mock()
        sftploader_mock.download_dir = 'foo'

        column_types = {'bar': np.float64}

        data_coupler.SFTPLoader.read_csv_into_dataframe(sftploader_mock, 'baz', column_types)

        pd_mock.assert_called_with(Path('foo', 'baz'), names=['bar'], dtype=column_types)

    def test_read_csv_into_dataframe_no_column_names(self, mocker):
        pd_mock = mocker.Mock()
        mocker.patch.object(pd, 'read_csv', new=pd_mock)

        sftploader_mock = mocker.Mock()
        sftploader_mock.download_dir = 'foo'

        data_coupler.SFTPLoader.read_csv_into_dataframe(sftploader_mock, 'baz')

        pd_mock.assert_called_with(Path('foo', 'baz'), names=None, dtype=None)
