from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas import testing as tm

import palletjack


class TestGSheetsLoader:

    def test_load_specific_worksheet_into_dataframe_by_id(self, mocker):
        sheet_mock = mocker.Mock()

        client_mock = mocker.Mock()
        client_mock.open_by_key.return_value = sheet_mock

        gsheet_loader_mock = mocker.Mock()
        gsheet_loader_mock.gsheets_client = client_mock

        palletjack.GSheetLoader.load_specific_worksheet_into_dataframe(gsheet_loader_mock, 'foobar', 5)

        sheet_mock.worksheet.assert_called_once_with('index', 5)
        sheet_mock.worksheet_by_title.assert_not_called()

    def test_load_specific_worksheet_into_dataframe_by_title(self, mocker):
        sheet_mock = mocker.Mock()

        client_mock = mocker.Mock()
        client_mock.open_by_key.return_value = sheet_mock

        gsheet_loader_mock = mocker.Mock()
        gsheet_loader_mock.gsheets_client = client_mock

        palletjack.GSheetLoader.load_specific_worksheet_into_dataframe(
            gsheet_loader_mock, 'foobar', '2015', by_title=True
        )

        sheet_mock.worksheet.assert_not_called
        sheet_mock.worksheet_by_title.assert_called_once_with('2015')

    def test_load_all_worksheets_into_dataframes_single_worksheet(self, mocker):
        worksheet_mock = mocker.Mock()
        worksheet_mock.title = 'ws1'
        worksheet_mock.get_as_df.return_value = 'df1'

        sheet_mock = mocker.Mock()
        sheet_mock.worksheets.return_value = [worksheet_mock]

        client_mock = mocker.Mock()
        client_mock.open_by_key.return_value = sheet_mock

        gsheet_loader_mock = mocker.Mock()
        gsheet_loader_mock.gsheets_client = client_mock

        df_dict = palletjack.GSheetLoader.load_all_worksheets_into_dataframes(gsheet_loader_mock, 'foobar')

        test_dict = {'ws1': 'df1'}

        assert df_dict == test_dict

    def test_load_all_worksheets_into_dataframes_multiple_worksheets(self, mocker):
        ws1_mock = mocker.Mock()
        ws1_mock.title = 'ws1'
        ws1_mock.get_as_df.return_value = 'df1'

        ws2_mock = mocker.Mock()
        ws2_mock.title = 'ws2'
        ws2_mock.get_as_df.return_value = 'df2'

        sheet_mock = mocker.Mock()
        sheet_mock.worksheets.return_value = [ws1_mock, ws2_mock]

        client_mock = mocker.Mock()
        client_mock.open_by_key.return_value = sheet_mock

        gsheet_loader_mock = mocker.Mock()
        gsheet_loader_mock.gsheets_client = client_mock

        df_dict = palletjack.GSheetLoader.load_all_worksheets_into_dataframes(gsheet_loader_mock, 'foobar')

        test_dict = {'ws1': 'df1', 'ws2': 'df2'}

        assert df_dict == test_dict

    def test_combine_worksheets_into_single_dataframe_combines_properly(self, mocker):
        df1 = pd.DataFrame({
            'foo': [1, 2],
            'bar': [3, 4],
        })

        df2 = pd.DataFrame({
            'foo': [10, 11],
            'bar': [12, 13],
        })

        df_dict = {'df1': df1, 'df2': df2}

        combined_df = palletjack.GSheetLoader.combine_worksheets_into_single_dataframe(mocker.Mock(), df_dict)

        test_df = pd.DataFrame({
            'worksheet': ['df1', 'df1', 'df2', 'df2'],
            'foo': [1, 2, 10, 11],
            'bar': [3, 4, 12, 13],
        })
        test_df.index = [0, 1, 0, 1]
        test_df.index.name = 'row'

        tm.assert_frame_equal(combined_df, test_df)

    def test_combine_worksheets_into_single_dataframe_raises_error_on_mismatched_columns(self, mocker):
        df1 = pd.DataFrame({
            'foo': [1, 2],
            'bar': [3, 4],
        })

        df2 = pd.DataFrame({
            'foo': [10, 11],
            'baz': [12, 13],
        })

        df_dict = {'df1': df1, 'df2': df2}

        with pytest.raises(ValueError, match='Columns do not match; cannot create mutli-index dataframe'):
            combined_df = palletjack.GSheetLoader.combine_worksheets_into_single_dataframe(mocker.Mock(), df_dict)


class TestGoogleDriveDownloader:

    def test_get_filename_from_response_works_normally(self, mocker):
        response = mocker.Mock()
        response.headers = {
            'foo': 'bar',
            'Content-Disposition': 'attachment;filename="file.name";filename*=UTF-8\'\'file.name'
        }

        filename = palletjack.GoogleDriveDownloader._get_filename_from_response(response)

        assert filename == 'file.name'

    def test_get_filename_from_response_raises_error_if_not_found(self, mocker):
        response = mocker.Mock()
        response.headers = {'foo': 'bar', 'Content-Disposition': 'attachment;filename*=UTF-8\'\'file.name'}

        with pytest.raises(ValueError, match='`filename=` not found in response header'):
            palletjack.GoogleDriveDownloader._get_filename_from_response(response)

    def test_get_file_info_works_normally(self, mocker):
        session_mock = mocker.Mock()
        mocker.patch('requests.Session', new=session_mock)

        palletjack.GoogleDriveDownloader._get_file_info('foo_file_id')

        assert session_mock.get.called_with('https://docs.google.com/uc?export=download', {'id': 'foo_file_id'}, True)

    def test_save_response_content_skips_empty_chunks(self, mocker):

        response_mock = mocker.MagicMock()
        response_mock.iter_content.return_value = [b'\x01', b'', b'\x02']

        open_mock = mocker.mock_open()
        mocker.patch('builtins.open', open_mock)

        palletjack.GoogleDriveDownloader._save_response_content(response_mock, '/foo/bar', chunk_size=1)

        assert open_mock().write.call_args_list[0][0] == (b'\x01',)
        assert open_mock().write.call_args_list[1][0] == (b'\x02',)

    def test_download_file_from_google_drive_creates_filename(self, mocker):

        mocker.patch.object(palletjack.GoogleDriveDownloader, '_get_file_info', return_value='response')
        mocker.patch.object(palletjack.GoogleDriveDownloader, '_get_filename_from_response', return_value='baz.png')
        save_mock = mocker.Mock()
        mocker.patch.object(palletjack.GoogleDriveDownloader, '_save_response_content', save_mock)

        downloader = palletjack.GoogleDriveDownloader('/foo/bar')

        downloader.download_file_from_google_drive('1234')

        save_mock.assert_called_with('response', Path('/foo/bar/baz.png'))


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
