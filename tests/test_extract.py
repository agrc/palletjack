import json
import logging
import re
from pathlib import Path

import arcgis
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import requests
import ujson
from pandas import testing as tm

from palletjack import extract


class TestGSheetsLoader:

    def test_load_specific_worksheet_into_dataframe_by_id(self, mocker):
        sheet_mock = mocker.Mock()

        client_mock = mocker.Mock()
        client_mock.open_by_key.return_value = sheet_mock

        gsheet_loader_mock = mocker.Mock()
        gsheet_loader_mock.gsheets_client = client_mock

        extract.GSheetLoader.load_specific_worksheet_into_dataframe(gsheet_loader_mock, 'foobar', 5)

        sheet_mock.worksheet.assert_called_once_with('index', 5)
        sheet_mock.worksheet_by_title.assert_not_called()

    def test_load_specific_worksheet_into_dataframe_by_title(self, mocker):
        sheet_mock = mocker.Mock()

        client_mock = mocker.Mock()
        client_mock.open_by_key.return_value = sheet_mock

        gsheet_loader_mock = mocker.Mock()
        gsheet_loader_mock.gsheets_client = client_mock

        extract.GSheetLoader.load_specific_worksheet_into_dataframe(gsheet_loader_mock, 'foobar', '2015', by_title=True)

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

        df_dict = extract.GSheetLoader.load_all_worksheets_into_dataframes(gsheet_loader_mock, 'foobar')

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

        df_dict = extract.GSheetLoader.load_all_worksheets_into_dataframes(gsheet_loader_mock, 'foobar')

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

        combined_df = extract.GSheetLoader.combine_worksheets_into_single_dataframe(mocker.Mock(), df_dict)

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

        with pytest.raises(ValueError, match='Columns do not match; cannot create multi-index dataframe'):
            combined_df = extract.GSheetLoader.combine_worksheets_into_single_dataframe(mocker.Mock(), df_dict)


class TestGoogleDriveDownloader:

    def test_get_filename_from_response_works_normally(self, mocker):
        response = mocker.Mock()
        response.headers = {
            'foo': 'bar',
            'Content-Disposition': 'attachment;filename="file.name";filename*=UTF-8\'\'file.name'
        }

        filename = extract.GoogleDriveDownloader._get_filename_from_response(response)

        assert filename == 'file.name'

    def test_get_filename_from_response_raises_error_if_not_found(self, mocker):
        response = mocker.Mock()
        response.headers = {'foo': 'bar', 'Content-Disposition': 'attachment'}

        with pytest.raises(ValueError, match='filename not found in response header'):
            extract.GoogleDriveDownloader._get_filename_from_response(response)

    def test_get_http_response_works_normally(self, mocker):
        response_mock = mocker.Mock()
        response_mock.headers = {'Content-Type': 'image/jpeg'}
        get_mock = mocker.Mock()
        get_mock.return_value = response_mock
        mocker.patch('requests.get', new=get_mock)

        extract.GoogleDriveDownloader._get_http_response(mocker.Mock(), 'foo_file_id')

        get_mock.assert_called_with(
            'https://docs.google.com/uc?export=download', params={'id': 'foo_file_id'}, stream=True
        )

    def test_get_http_response_raises_error_on_text_response(self, mocker):
        response_mock = mocker.Mock()
        response_mock.headers = {'Content-Type': 'text/html'}
        get_mock = mocker.Mock()
        get_mock.return_value = response_mock
        mocker.patch('requests.get', new=get_mock)

        with pytest.raises(RuntimeError) as error:
            extract.GoogleDriveDownloader._get_http_response(mocker.Mock(), 'foo_file_id')
        assert 'Cannot access foo_file_id (is it publicly shared?). Response header in log.' in str(error.value)

    def test_save_response_content_skips_empty_chunks(self, mocker):

        response_mock = mocker.MagicMock()
        response_mock.iter_content.return_value = [b'\x01', b'', b'\x02']

        open_mock = mocker.mock_open()
        mocker.patch('pathlib.Path.open', open_mock)

        extract.GoogleDriveDownloader._save_response_content(response_mock, Path('/foo/bar'), chunk_size=1)

        assert open_mock().write.call_args_list[0][0] == (b'\x01',)
        assert open_mock().write.call_args_list[1][0] == (b'\x02',)

    def test_download_file_from_google_drive_creates_filename(self, mocker):

        mocker.patch.object(extract.GoogleDriveDownloader, '_get_file_id_from_sharing_link')
        mocker.patch.object(extract.GoogleDriveDownloader, '_get_http_response', return_value='response')
        mocker.patch.object(extract.GoogleDriveDownloader, '_get_filename_from_response', return_value='baz.png')
        save_mock = mocker.Mock()
        mocker.patch.object(extract.GoogleDriveDownloader, '_save_response_content', save_mock)

        downloader = extract.GoogleDriveDownloader('/foo/bar')

        downloader.download_file_from_google_drive('1234', 42)

        save_mock.assert_called_with('response', Path('/foo/bar/baz.png'))

    def test_download_file_from_google_drive_handles_empty_string_link(self, mocker, caplog):

        file_id_mock = mocker.Mock()
        mocker.patch.object(extract.GoogleDriveDownloader, '_get_file_id_from_sharing_link', file_id_mock)

        downloader = extract.GoogleDriveDownloader('/foo/bar')

        caplog.set_level(logging.DEBUG, logger='palletjack.extract.GoogleDriveDownloader')
        caplog.clear()
        result = downloader.download_file_from_google_drive('', 42)

        file_id_mock.assert_not_called()
        assert ['Row 42 has no attachment info'] == [rec.message for rec in caplog.records]
        assert result is None

    def test_download_file_from_google_drive_handles_None_link(self, mocker, caplog):

        file_id_mock = mocker.Mock()
        mocker.patch.object(extract.GoogleDriveDownloader, '_get_file_id_from_sharing_link', file_id_mock)

        downloader = extract.GoogleDriveDownloader('/foo/bar')

        caplog.set_level(logging.DEBUG, logger='palletjack.extract.GoogleDriveDownloader')
        caplog.clear()
        result = downloader.download_file_from_google_drive(None, 42)

        file_id_mock.assert_not_called()
        assert ['Row 42 has no attachment info'] == [rec.message for rec in caplog.records]
        assert result is None

    def test_download_file_from_google_drive_handles_download_error(self, mocker, caplog):

        mocker.patch.object(extract.GoogleDriveDownloader, '_get_file_id_from_sharing_link', return_value='google_id')
        mocker.patch.object(extract.GoogleDriveDownloader, '_get_http_response', side_effect=Exception('Boom'))
        filename_mock = mocker.Mock()
        mocker.patch.object(extract.GoogleDriveDownloader, '_get_filename_from_response', new=filename_mock)
        # save_mock = mocker.Mock()
        # mocker.patch.object(extract.GoogleDriveDownloader, '_save_response_content', save_mock)

        downloader = extract.GoogleDriveDownloader('/foo/bar')

        caplog.set_level(logging.DEBUG, logger='palletjack.extract.GoogleDriveDownloader')
        caplog.clear()
        result = downloader.download_file_from_google_drive('1234', 42)

        filename_mock.assert_not_called()
        assert [
            'Row 42: downloading shared file 1234',
            'Row 42: extracted file id google_id',
            'Row 42: Couldn\'t download 1234',
            'Boom',
        ] == [rec.message for rec in caplog.records]
        assert result is None

    def test_download_file_from_google_drive_pauses_appropriately(self, mocker):
        sleep_mock = mocker.patch.object(extract, 'sleep')
        mocker.patch.object(extract.GoogleDriveDownloader, '_get_file_id_from_sharing_link')
        mocker.patch.object(extract.GoogleDriveDownloader, '_get_http_response', return_value='response')
        mocker.patch.object(extract.GoogleDriveDownloader, '_get_filename_from_response', return_value='baz.png')
        mocker.patch.object(extract.GoogleDriveDownloader, '_save_response_content')

        downloader = extract.GoogleDriveDownloader('/foo/bar')

        downloader.download_file_from_google_drive('1234', 42, 1.5)

        sleep_mock.assert_called_with(1.5)
        sleep_mock.assert_called_once()

    def test_download_attachments_from_dataframe_sleeps_specified_time(self, mocker):

        sleep_mock = mocker.patch.object(extract, 'sleep')
        mocker.patch.object(extract.GoogleDriveDownloader, '_get_file_id_from_sharing_link')
        mocker.patch.object(extract.GoogleDriveDownloader, '_get_http_response', return_value='response')
        mocker.patch.object(extract.GoogleDriveDownloader, '_get_filename_from_response', return_value='baz.png')
        mocker.patch.object(extract.GoogleDriveDownloader, '_save_response_content')

        downloader = extract.GoogleDriveDownloader('/foo/bar')

        sheet_dataframe = pd.DataFrame({
            'join_id': [1],
            'link': ['a'],
        })

        downloader.download_attachments_from_dataframe(sheet_dataframe, 'link', 'join_id', 'path')

        sleep_mock.assert_called_with(5)
        sleep_mock.assert_called_once()

    def test_download_attachments_from_dataframe_handles_multiple_rows(self, mocker):

        downloader_mock = mocker.Mock()
        downloader_mock.download_file_from_google_drive.side_effect = ['foo/bar.png', 'baz/boo.png']

        sheet_dataframe = pd.DataFrame({
            'join_id': [1, 2],
            'link': ['a', 'b'],
        })

        downloaded_df = extract.GoogleDriveDownloader.download_attachments_from_dataframe(
            downloader_mock, sheet_dataframe, 'link', 'join_id', 'path'
        )

        test_df = pd.DataFrame({
            'join_id': [1, 2],
            'link': ['a', 'b'],
            'path': ['foo/bar.png', 'baz/boo.png'],
        })

        tm.assert_frame_equal(downloaded_df, test_df)

    def test_get_file_id_from_sharing_link_extracts_group_file_link(self, mocker):

        downloader = extract.GoogleDriveDownloader('foo')
        test_link = 'https://drive.google.com/file/d/abcdefghijklm_opqrstuvwxy/view?usp=sharing'

        file_id = downloader._get_file_id_from_sharing_link(test_link)

        assert file_id == 'abcdefghijklm_opqrstuvwxy'

    def test_get_file_id_from_sharing_link_extracts_group_file_link_with_dashes(self, mocker):

        downloader = extract.GoogleDriveDownloader('foo')
        test_link = 'https://drive.google.com/file/d/abcdefghijklm-opqrstuvwxy/view?usp=sharing'

        file_id = downloader._get_file_id_from_sharing_link(test_link)

        assert file_id == 'abcdefghijklm-opqrstuvwxy'

    def test_get_file_id_from_sharing_link_extracts_group_open_link(self, mocker):

        downloader = extract.GoogleDriveDownloader('foo')
        test_link = 'https://drive.google.com/open?id=abcdefghijklm_opqrstuvwxy'

        file_id = downloader._get_file_id_from_sharing_link(test_link)

        assert file_id == 'abcdefghijklm_opqrstuvwxy'

    def test_get_file_id_from_sharing_link_raises_error_on_failed_match(self, mocker):

        downloader = extract.GoogleDriveDownloader('foo')
        test_link = 'bad_link'

        with pytest.raises(RuntimeError, match='Regex could not match sharing link bad_link'):
            file_id = downloader._get_file_id_from_sharing_link(test_link)


class TestGoogleDriveDownloaderAPI:

    def test_get_request_and_filename_from_drive_api_returns_both(self, mocker):
        client = mocker.Mock()
        client.drive.service.files.return_value.get_media.return_value = 'request'
        client.drive.service.files.return_value.get.return_value.execute.return_value = {'name': 'foo.bar'}

        request, filename = extract.GoogleDriveDownloader._get_request_and_filename_from_drive_api(
            mocker.Mock(), client, 'bar'
        )

        assert request == 'request'
        assert filename == 'foo.bar'

    def test_get_request_and_filename_from_drive_api_adds_extension(self, mocker):
        client = mocker.Mock()
        client.drive.service.files.return_value.get_media.return_value = 'request'
        client.drive.service.files.return_value.get.return_value.execute.return_value = {
            'name': 'foo',
            'mimeType': 'image/jpeg'
        }

        request, filename = extract.GoogleDriveDownloader._get_request_and_filename_from_drive_api(
            mocker.Mock(), client, 'bar'
        )

        assert request == 'request'
        assert filename == 'foo.jpg'

    def test_get_request_and_filename_from_drive_api_warns_on_missing_mimeType(self, mocker, caplog):
        client = mocker.Mock()
        client.drive.service.files.return_value.get_media.return_value = 'request'
        client.drive.service.files.return_value.get.return_value.execute.return_value = {
            'name': 'foo',
        }

        downloader = extract.GoogleDriveDownloader('/foo/bar')

        caplog.set_level(logging.DEBUG, logger='palletjack.loaders.GoogleDriveDownloader')
        caplog.clear()

        request, filename = downloader._get_request_and_filename_from_drive_api(client, 'bar')

        assert ['bar: No MIME type in drive info, file extension not set'] == [rec.message for rec in caplog.records]
        assert request == 'request'
        assert filename == 'foo'

    def test_get_request_and_filename_from_drive_api_warns_on_cant_find_extension(self, mocker, caplog):
        client = mocker.Mock()
        client.drive.service.files.return_value.get_media.return_value = 'request'
        client.drive.service.files.return_value.get.return_value.execute.return_value = {
            'name': 'foo',
            'mimeType': 'bee/boo'
        }

        downloader = extract.GoogleDriveDownloader('/foo/bar')

        caplog.set_level(logging.DEBUG, logger='palletjack.loaders.GoogleDriveDownloader')
        caplog.clear()

        request, filename = downloader._get_request_and_filename_from_drive_api(client, 'bar')

        assert ['bar: Unable to determine file extension from MIME type, file extension not set'] == \
            [rec.message for rec in caplog.records]
        assert request == 'request'
        assert filename == 'foo'

    def test_save_get_media_content_works_multiple_chunks(self, mocker):
        mocker.patch.object(extract, 'BytesIO')
        downloader_mock = mocker.patch.object(extract, 'MediaIoBaseDownload')
        downloader_mock.return_value.next_chunk.side_effect = [(1, False), (2, True)]

        extract.GoogleDriveDownloader._save_get_media_content(mocker.Mock(), 'foo', mocker.Mock())

        assert downloader_mock.return_value.next_chunk.call_count == 2

    def test_download_file_from_google_drive_using_api_creates_filename(self, mocker):

        mocker.patch.object(extract.GoogleDriveDownloader, '_get_file_id_from_sharing_link')
        mocker.patch.object(
            extract.GoogleDriveDownloader,
            '_get_request_and_filename_from_drive_api',
            return_value=('response', 'baz.png')
        )
        save_mock = mocker.patch.object(extract.GoogleDriveDownloader, '_save_get_media_content')

        downloader = extract.GoogleDriveDownloader('/foo/bar')

        downloader.download_file_from_google_drive_using_api(mocker.Mock(), '1234', 42)

        save_mock.assert_called_with('response', Path('/foo/bar/baz.png'))

    def test_download_file_from_google_drive_using_api_handles_empty_string_link(self, mocker, caplog):

        file_id_mock = mocker.patch.object(extract.GoogleDriveDownloader, '_get_file_id_from_sharing_link')

        downloader = extract.GoogleDriveDownloader('/foo/bar')

        caplog.set_level(logging.DEBUG, logger='palletjack.extract.GoogleDriveDownloader')
        caplog.clear()
        result = downloader.download_file_from_google_drive_using_api(mocker.Mock(), '', 42)

        file_id_mock.assert_not_called()
        assert ['Row 42 has no attachment info'] == [rec.message for rec in caplog.records]
        assert result is None

    def test_download_file_from_google_drive_using_api_handles_None_link(self, mocker, caplog):

        file_id_mock = mocker.patch.object(extract.GoogleDriveDownloader, '_get_file_id_from_sharing_link')

        downloader = extract.GoogleDriveDownloader('/foo/bar')

        caplog.set_level(logging.DEBUG, logger='palletjack.extract.GoogleDriveDownloader')
        caplog.clear()
        result = downloader.download_file_from_google_drive_using_api(mocker.Mock(), None, 42)

        file_id_mock.assert_not_called()
        assert ['Row 42 has no attachment info'] == [rec.message for rec in caplog.records]
        assert result is None

    def test_download_file_from_google_drive_using_api_handles_download_error(self, mocker, caplog):

        mocker.patch.object(extract.GoogleDriveDownloader, '_get_file_id_from_sharing_link', return_value='google_id')
        mocker.patch.object(extract.utils, 'sleep')
        mocker.patch.object(
            extract.GoogleDriveDownloader, '_get_request_and_filename_from_drive_api', side_effect=Exception('Boom')
        )

        saver_mock = mocker.patch.object(extract.GoogleDriveDownloader, '_save_get_media_content')

        downloader = extract.GoogleDriveDownloader('/foo/bar')

        caplog.set_level(logging.DEBUG, logger='palletjack.extract.GoogleDriveDownloader')
        caplog.clear()
        result = downloader.download_file_from_google_drive_using_api(mocker.Mock(), '1234', 42)

        saver_mock.assert_not_called()
        assert [
            'Row 42: downloading file 1234',
            'Row 42: extracted file id google_id',
            'Row 42: Couldn\'t download 1234',
            'Boom',
        ] == [rec.message for rec in caplog.records]
        assert result is None

    def test_download_attachments_from_dataframe_using_api_handles_multiple_rows(self, mocker):

        mocker.patch.object(extract.utils, 'authorize_pygsheets')

        downloader_mock = mocker.Mock()
        downloader_mock.download_file_from_google_drive_using_api.side_effect = ['foo/bar.png', 'baz/boo.png']

        sheet_dataframe = pd.DataFrame({
            'join_id': [1, 2],
            'link': ['a', 'b'],
        })

        downloaded_df = extract.GoogleDriveDownloader.download_attachments_from_dataframe_using_api(
            downloader_mock, 'service_file', sheet_dataframe, 'link', 'join_id', 'path'
        )

        test_df = pd.DataFrame({
            'join_id': [1, 2],
            'link': ['a', 'b'],
            'path': ['foo/bar.png', 'baz/boo.png'],
        })

        tm.assert_frame_equal(downloaded_df, test_df)


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

        extract.SFTPLoader.download_sftp_folder_contents(sftploader_mock)

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

        extract.SFTPLoader.download_sftp_single_file(sftploader_mock, 'filename', 'upload')

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

        extract.SFTPLoader.read_csv_into_dataframe(sftploader_mock, 'baz', column_types)

        pd_mock.assert_called_with(Path('foo', 'baz'), names=['bar'], dtype=column_types)

    def test_read_csv_into_dataframe_no_column_names(self, mocker):
        pd_mock = mocker.Mock()
        pd_mock.return_value = pd.DataFrame()
        mocker.patch.object(pd, 'read_csv', new=pd_mock)

        sftploader_mock = mocker.Mock()
        sftploader_mock.download_dir = 'foo'

        extract.SFTPLoader.read_csv_into_dataframe(sftploader_mock, 'baz')

        pd_mock.assert_called_with(Path('foo', 'baz'), names=None, dtype=None)


class TestPostgresLoader:

    def test_get_postgres_connection(self, mocker):

        mocker.patch.object(
            gpd, 'read_postgis', return_value=gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        )

        loader = extract.PostgresLoader('host', 'app', 'user', 'password')
        dataframe = loader.read_table_into_dataframe('table', 'name', '4326', 'geometry')

        assert dataframe.spatial.geometry_type == ['polygon']


class TestRESTServiceLoader:

    def test_get_features_gets_max_record_count_from_properties(self, mocker):

        layer_mock = mocker.patch('palletjack.extract._ServiceLayer').return_value
        layer_mock.max_record_count = 42
        layer_mock.get_object_ids.return_value = list(range(0, 142))

        mocker.patch('palletjack.extract.pd.concat')
        mocker.patch('palletjack.extract.time.sleep')
        chunker_mock = mocker.patch('palletjack.utils.chunker')

        extract.RESTServiceLoader.get_features(mocker.Mock(), chunk_size=None)

        chunker_mock.assert_called_once_with(list(range(0, 142)), 42)

    def test_get_features_chunks_smaller_final_chunk(self, mocker):
        layer_mock = mocker.patch('palletjack.extract._ServiceLayer').return_value
        layer_mock.max_record_count = 100
        layer_mock.oid_field = 'OBJECTID'
        layer_mock.get_object_ids.return_value = list(range(0, 142))

        mocker.patch('palletjack.extract.pd.concat')
        mocker.patch('palletjack.extract.time.sleep')

        extract.RESTServiceLoader.get_features(mocker.Mock())

        assert layer_mock.get_unique_id_list_as_dataframe.call_args_list == [
            mocker.call('OBJECTID', list(range(0, 100))),
            mocker.call('OBJECTID', list(range(100, 142))),
        ]

    def test_get_features_chunks_single_chunk_smaller_than_max_record_count(self, mocker):
        layer_mock = mocker.patch('palletjack.extract._ServiceLayer').return_value
        layer_mock.max_record_count = 10
        layer_mock.oid_field = 'OBJECTID'
        layer_mock.get_object_ids.return_value = [10, 11, 12, 13, 14]

        mocker.patch('palletjack.extract.pd.concat', return_value=pd.DataFrame([0, 1, 2, 3, 4]))
        mocker.patch('palletjack.extract.time.sleep')

        extract.RESTServiceLoader.get_features(mocker.Mock(), 100)

        layer_mock.get_unique_id_list_as_dataframe.assert_called_once_with('OBJECTID', [10, 11, 12, 13, 14])

    def test_get_features_sleeps(self, mocker):
        layer_mock = mocker.patch('palletjack.extract._ServiceLayer').return_value
        layer_mock.get_object_ids.return_value = [10, 11, 12, 13, 14]

        mocker.patch('palletjack.extract.pd.concat', return_value=pd.DataFrame([0, 1, 2, 3, 4]))
        sleep_mock = mocker.patch('palletjack.extract.time.sleep')
        mocker.patch('palletjack.extract.random.randint', return_value=42)

        extract.RESTServiceLoader.get_features(mocker.Mock(), 100)

        sleep_mock.assert_called_once_with(.42)

    def test_get_features_warns_on_zero_oids(self, mocker):
        layer_mock = mocker.patch('palletjack.extract._ServiceLayer').return_value
        layer_mock.get_object_ids.return_value = []
        layer_mock.layer_url = 'foo.bar/0'

        with pytest.warns(UserWarning, match='Layer foo.bar/0 has no features'):
            features = extract.RESTServiceLoader.get_features(mocker.Mock(), 100)
        assert features is None

    def test_init_strips_trailing_slash(self):
        test_loader = extract.RESTServiceLoader('foo.bar/')

        assert test_loader.url == 'foo.bar'

    def test_init_doesnt_strip_trailing_slash_if_no_trailing_slash(self):
        test_loader = extract.RESTServiceLoader('foo.bar')

        assert test_loader.url == 'foo.bar'

    def test_get_feature_layers_info_only_returns_feature_layers_and_tables(self, mocker):
        response_json = {
            'layers': [{
                'id': 0,
                'type': 'Feature Layer'
            }, {
                'id': 1,
                'type': 'Group Layer'
            }, {
                'id': 2,
                'type': 'Table'
            }]
        }
        expected_layers = [{'id': 0, 'type': 'Feature Layer'}, {'id': 2, 'type': 'Table'}]

        mocker.patch('palletjack.extract.utils.retry', return_value=mocker.Mock(json=lambda: response_json))

        assert extract.RESTServiceLoader.get_feature_layers_info(mocker.Mock()) == expected_layers

    def test_get_feature_layers_info_returns_empty_list_if_no_feature_layers(self, mocker):
        response_json = {'layers': [{'id': 0, 'type': 'Group Layer'}]}

        mocker.patch('palletjack.extract.utils.retry', return_value=mocker.Mock(json=lambda: response_json))

        assert extract.RESTServiceLoader.get_feature_layers_info(mocker.Mock()) == []

    def test_get_feature_layers_info_raises_on_json_parse_error(self, mocker):
        response_mock = mocker.Mock()
        response_mock.json.side_effect = json.JSONDecodeError('foo', 'bar', 42)

        mocker.patch('palletjack.extract.utils.retry', return_value=response_mock)

        with pytest.raises(RuntimeError, match=re.escape('Could not parse response from foo.bar')):
            extract.RESTServiceLoader.get_feature_layers_info(mocker.Mock(url='foo.bar'))

    def test_get_feature_layers_info_raises_on_missing_layers_key(self, mocker):
        mocker.patch('palletjack.extract.utils.retry', return_value=mocker.Mock(json=lambda: {'foo': 'bar'}))

        with pytest.raises(RuntimeError, match=re.escape('Response from foo.bar does not contain layer information')):
            extract.RESTServiceLoader.get_feature_layers_info(mocker.Mock(url='foo.bar'))

    def test_get_feature_layers_info_raises_on_missing_layer_type(self, mocker):

        mocker.patch('palletjack.extract.utils.retry', return_value=mocker.Mock(json=lambda: {'layers': [{'id': 0}]}))

        with pytest.raises(RuntimeError, match=re.escape('Layer info did not contain layer type')):
            extract.RESTServiceLoader.get_feature_layers_info(mocker.Mock(url='foo.bar'))

    def test__get_service_info_uses_proper_query(self, mocker):
        mock_response = mocker.Mock()
        mock_response.status_code = 200

        get_mock = mocker.patch('palletjack.extract.requests.get')
        get_mock.return_value = mock_response

        service_mock = mocker.Mock(url='http://fakeurl.com', timeout=10)

        response = extract.RESTServiceLoader._get_service_info(service_mock)

        get_mock.assert_called_once_with('http://fakeurl.com/query', params={'f': 'json'}, timeout=10)
        assert response == mock_response

    def test__get_service_info_raises_http_error(self, mocker):
        mock_response = mocker.Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError

        get_mock = mocker.patch('palletjack.extract.requests.get')
        get_mock.return_value = mock_response

        service_mock = mocker.Mock(url='http://fakeurl.com', timeout=10)

        with pytest.raises(requests.exceptions.HTTPError):
            extract.RESTServiceLoader._get_service_info(service_mock)


class Test_ServiceLayer:

    def test_get_layer_info_works_normally(self, mocker):

        retry_mock = mocker.patch('palletjack.extract.utils.retry')
        retry_mock.return_value.json.return_value = {
            'capabilities': 'foo,bar,baz',
            'type': 'Feature Layer',
            'maxRecordCount': 42
        }

        class_mock = mocker.Mock()
        class_mock.layer_url = 'foo.bar'
        class_mock.timeout = 5

        layer_info = extract._ServiceLayer._get_layer_info(class_mock)

        assert layer_info == {'capabilities': 'foo,bar,baz', 'type': 'Feature Layer', 'maxRecordCount': 42}

    def test_get_layer_info_raises_on_missing_capabilities(self, mocker):
        response_mock = mocker.Mock()
        response_mock.json.return_value = {'type': 'Feature Layer', 'maxRecordCount': 42}
        get_mock = mocker.patch('palletjack.extract.requests.get', return_value=response_mock)

        class_mock = mocker.Mock()
        class_mock.layer_url = 'foo.bar'
        class_mock.timeout = 5

        with pytest.raises(
            RuntimeError, match='Response does not contain layer information; ensure URL points to a valid layer'
        ):
            layer_info = extract._ServiceLayer._get_layer_info(class_mock)

    def test_get_layer_info_raises_on_missing_type(self, mocker):
        response_mock = mocker.Mock()
        response_mock.json.return_value = {'capabilities': 'foo,bar,baz', 'maxRecordCount': 42}
        get_mock = mocker.patch('palletjack.extract.requests.get', return_value=response_mock)

        class_mock = mocker.Mock()
        class_mock.layer_url = 'foo.bar'
        class_mock.timeout = 5

        with pytest.raises(
            RuntimeError, match='Response does not contain layer information; ensure URL points to a valid layer'
        ):
            layer_info = extract._ServiceLayer._get_layer_info(class_mock)

    def test_get_layer_info_raises_on_missing_record_count(self, mocker):
        response_mock = mocker.Mock()
        response_mock.json.return_value = {'capabilities': 'foo,bar,baz', 'type': 'Feature Layer'}
        get_mock = mocker.patch('palletjack.extract.requests.get', return_value=response_mock)

        class_mock = mocker.Mock()
        class_mock.layer_url = 'foo.bar'
        class_mock.timeout = 5

        with pytest.raises(
            RuntimeError, match='Response does not contain layer information; ensure URL points to a valid layer'
        ):
            layer_info = extract._ServiceLayer._get_layer_info(class_mock)

    def test_get_layer_info_retries_on_timeout(self, mocker):
        mocker.patch.object(extract.utils, 'sleep')
        response_mock = mocker.Mock()
        response_mock.json.return_value = {'capabilities': 'foo,bar,baz', 'type': 'Feature Layer', 'maxRecordCount': 42}
        get_mock = mocker.patch(
            'palletjack.extract.requests.get', side_effect=[requests.exceptions.Timeout, response_mock]
        )

        class_mock = mocker.Mock()
        class_mock.layer_url = 'foo.bar'
        class_mock.timeout = 5

        extract._ServiceLayer._get_layer_info(class_mock)

        assert get_mock.call_count == 2
        get_mock.assert_called_with('foo.bar', params={'f': 'json'}, timeout=5)

    def test_get_object_ids_works_normally(self, mocker):
        response_mock = mocker.Mock()
        response_mock.json.return_value = {'objectIds': [8, 16, 42]}
        get_mock = mocker.patch('palletjack.extract.requests.get', return_value=response_mock)

        class_mock = mocker.Mock()
        class_mock.layer_url = 'foo.bar'
        class_mock.timeout = 5
        class_mock.envelope_params = None
        class_mock.where_clause = '1=1'

        record_count = extract._ServiceLayer.get_object_ids(class_mock)

        assert record_count == [8, 16, 42]
        get_mock.assert_called_once_with(
            'foo.bar/query', params={
                'returnIdsOnly': 'true',
                'f': 'json',
                'where': '1=1'
            }, timeout=5
        )

    def test_get_object_ids_includes_envelope_params(self, mocker):
        expected_params = {
            'returnIdsOnly': 'true',
            'f': 'json',
            'where': '1=1',
            'geometry': 'envelope',
            'geometryType': 'esriGeometryEnvelope',
            'inSR': 'sr'
        }

        response_mock = mocker.Mock()
        response_mock.json.return_value = {'objectIds': [8, 16, 42]}
        get_mock = mocker.patch('palletjack.extract.requests.get', return_value=response_mock)

        class_mock = mocker.Mock()
        class_mock.layer_url = 'foo.bar'
        class_mock.timeout = 5
        class_mock.envelope_params = {'geometry': 'envelope', 'geometryType': 'esriGeometryEnvelope', 'inSR': 'sr'}
        class_mock.where_clause = '1=1'
        record_count = extract._ServiceLayer.get_object_ids(class_mock)

        assert record_count == [8, 16, 42]
        get_mock.assert_called_once_with('foo.bar/query', params=expected_params, timeout=5)

    def test_get_object_ids_includes_where_clause(self, mocker):
        expected_params = {'returnIdsOnly': 'true', 'f': 'json', 'where': 'foo = bar'}

        response_mock = mocker.Mock()
        response_mock.json.return_value = {'objectIds': [8, 16, 42]}
        get_mock = mocker.patch('palletjack.extract.requests.get', return_value=response_mock)

        class_mock = mocker.Mock()
        class_mock.layer_url = 'foo.bar'
        class_mock.timeout = 5
        class_mock.envelope_params = None
        class_mock.where_clause = 'foo = bar'
        record_count = extract._ServiceLayer.get_object_ids(class_mock)

        assert record_count == [8, 16, 42]
        get_mock.assert_called_once_with('foo.bar/query', params=expected_params, timeout=5)

    def test_get_object_ids_returns_sorted_ids(self, mocker):
        expected_ids = [8, 16, 42]
        input_ids = [42, 8, 16]

        response_mock = mocker.Mock()
        response_mock.json.return_value = {'objectIds': input_ids}
        get_mock = mocker.patch('palletjack.extract.requests.get', return_value=response_mock)

        class_mock = mocker.Mock()
        class_mock.layer_url = 'foo.bar'
        class_mock.timeout = 5
        class_mock.envelope_params = None
        class_mock.where_clause = '1=1'

        record_count = extract._ServiceLayer.get_object_ids(class_mock)

        assert record_count == expected_ids
        get_mock.assert_called_once_with(
            'foo.bar/query', params={
                'returnIdsOnly': 'true',
                'f': 'json',
                'where': '1=1'
            }, timeout=5
        )

    def test_get_object_ids_raises_on_key_error(self, mocker):
        response_mock = mocker.Mock()
        response_mock.json.return_value = {'foo': 'bar'}
        get_mock = mocker.patch('palletjack.extract.requests.get', return_value=response_mock)

        class_mock = mocker.Mock()
        class_mock.layer_url = 'foo.bar'
        class_mock.timeout = 5
        class_mock.envelope_params = None
        class_mock.where_clause = '1=1'

        with pytest.raises(RuntimeError, match=re.escape(f'Could not get object IDs from foo.bar')):
            record_count = extract._ServiceLayer.get_object_ids(class_mock)

    def test_get_object_ids_returns_empty_list_on_no_oids(self, mocker):
        response_mock = mocker.Mock()
        response_mock.json.return_value = {'objectIds': None}
        get_mock = mocker.patch('palletjack.extract.requests.get', return_value=response_mock)

        class_mock = mocker.Mock()
        class_mock.layer_url = 'foo.bar'
        class_mock.timeout = 5
        class_mock.envelope_params = None
        class_mock.where_clause = '1=1'

        record_count = extract._ServiceLayer.get_object_ids(class_mock)

        assert record_count == []
        get_mock.assert_called_once_with(
            'foo.bar/query', params={
                'returnIdsOnly': 'true',
                'f': 'json',
                'where': '1=1'
            }, timeout=5
        )

    def test_get_unique_id_list_as_dataframe_creates_list(self, mocker):
        class_mock = mocker.Mock()
        class_mock.layer_url = 'foo.bar'
        class_mock.timeout = 5
        class_mock.feature_params = {'outFields': '*', 'returnGeometry': 'true'}

        response_mock = mocker.Mock()
        response_mock.status_code = 200
        requests_mock = mocker.patch('palletjack.extract.requests.get', return_value=response_mock)

        mocker.patch('palletjack.extract.arcgis.features.FeatureSet')
        mocker.patch('palletjack.extract.len', return_value=5)

        oid_list = [10, 11, 12, 13, 14]

        extract._ServiceLayer.get_unique_id_list_as_dataframe(class_mock, 'OBJECTID', oid_list)

        requests_mock.assert_called_once_with(
            'foo.bar/query',
            params={
                'f': 'json',
                'outFields': '*',
                'returnGeometry': 'true',
                'where': 'OBJECTID in (10,11,12,13,14)'
            },
            timeout=5
        )

    def test_get_unique_id_list_as_dataframe_raises_on_404(self, mocker):
        class_mock = mocker.Mock()
        class_mock.layer_url = 'foo.bar'
        class_mock.timeout = 5
        class_mock.feature_params = {'outFields': '*', 'returnGeometry': 'true'}

        response_mock = mocker.Mock()
        response_mock.status_code = 404
        requests_mock = mocker.patch('palletjack.extract.requests.get', return_value=response_mock)

        oid_list = [10, 11, 12, 13, 14]

        with pytest.raises(RuntimeError, match=re.escape('Bad chunk response HTTP status code (404)')):
            extract._ServiceLayer.get_unique_id_list_as_dataframe(class_mock, 'OBJECTID', oid_list)

    def test_get_unique_id_list_as_dataframe_raises_on_json_error(self, mocker):
        class_mock = mocker.Mock()
        class_mock.layer_url = 'foo.bar'
        class_mock.timeout = 5
        class_mock.feature_params = {'outFields': '*', 'returnGeometry': 'true'}

        response_mock = mocker.Mock()
        response_mock.status_code = 200
        requests_mock = mocker.patch('palletjack.extract.requests.get', return_value=response_mock)

        mocker.patch(
            'palletjack.extract.arcgis.features.FeatureSet.from_json',
            side_effect=ujson.JSONDecodeError('foo', 'bar', 0)
        )

        oid_list = [10, 11, 12, 13, 14]

        with pytest.raises(RuntimeError, match=re.escape('Could not parse chunk features from response')):
            extract._ServiceLayer.get_unique_id_list_as_dataframe(class_mock, 'OBJECTID', oid_list)

    def test_get_unique_id_list_as_dataframe_raises_on_len_mismatch(self, mocker):
        class_mock = mocker.Mock()
        class_mock.layer_url = 'foo.bar'
        class_mock.timeout = 5
        class_mock.feature_params = {'outFields': '*', 'returnGeometry': 'true'}

        response_mock = mocker.Mock()
        response_mock.status_code = 200
        requests_mock = mocker.patch('palletjack.extract.requests.get', return_value=response_mock)

        featureset_mock = mocker.patch('palletjack.extract.arcgis.features.FeatureSet')
        featureset_mock.from_json.return_value.sdf.sort_values.return_value = pd.DataFrame({
            'foo': ['10', '11', '12', '13']
        })

        oid_list = [10, 11, 12, 13, 14]

        with pytest.raises(
            RuntimeError, match=re.escape('Missing features. 5 OIDs requested, but 4 features downloaded')
        ):
            extract._ServiceLayer.get_unique_id_list_as_dataframe(class_mock, 'OBJECTID', oid_list)

    def test_init_builds_envelope_params(self, mocker):
        mocker.patch('palletjack.extract._ServiceLayer._get_layer_info', return_value={'maxRecordCount': 8})
        mocker.patch('palletjack.extract._ServiceLayer._get_object_id_field')
        mocker.patch('palletjack.extract._ServiceLayer._check_layer_type')

        test_loader = extract._ServiceLayer(mocker.Mock(), 42, envelope_params={'geometry': 'eggs', 'inSR': 'spam'})

        assert test_loader.envelope_params == {
            'geometryType': 'esriGeometryEnvelope',
            'geometry': 'eggs',
            'inSR': 'spam'
        }

    def test_init_builds_feature_params(self, mocker):
        mocker.patch('palletjack.extract._ServiceLayer._get_layer_info', return_value={'maxRecordCount': 8})
        mocker.patch('palletjack.extract._ServiceLayer._get_object_id_field')
        mocker.patch('palletjack.extract._ServiceLayer._check_layer_type')

        test_loader = extract._ServiceLayer(
            mocker.Mock(), 42, feature_params={
                'where': 'eggs',
                'returnGeometry': 'spam'
            }
        )

        assert test_loader.feature_params == {'outFields': '*', 'where': 'eggs', 'returnGeometry': 'spam'}

    def test_init_raises_on_missing_envelope_geometry(self, mocker):

        with pytest.raises(
            ValueError, match='envelope_params must contain both the envelope geometry and its spatial reference'
        ):
            test_loader = extract._ServiceLayer.__init__(
                mocker.Mock(_get_layer_info=lambda: {'maxRecordCount': 8}),
                mocker.Mock(),
                42,
                envelope_params={'inSR': 'eggs'}
            )

    def test_init_raises_on_missing_envelope_sr(self, mocker):

        with pytest.raises(
            ValueError, match='envelope_params must contain both the envelope geometry and its spatial reference'
        ):
            test_loader = extract._ServiceLayer.__init__(
                mocker.Mock(_get_layer_info=lambda: {'maxRecordCount': 8}),
                mocker.Mock(),
                42,
                envelope_params={'geometry': 'eggs'}
            )

    def test_init_raises_on_missing_envelope_geometry_and_sr(self, mocker):

        with pytest.raises(
            ValueError, match='envelope_params must contain both the envelope geometry and its spatial reference'
        ):
            test_loader = extract._ServiceLayer.__init__(
                mocker.Mock(_get_layer_info=lambda: {'maxRecordCount': 8}),
                mocker.Mock(),
                42,
                envelope_params={'ham': 'eggs'}
            )

    def test_check_capabilities_raises_on_missing(self, mocker):
        response_json = {'capabilities': 'map,data'}

        with pytest.raises(
            RuntimeError, match=re.escape('query capability not in layer\'s capabilities ([\'map\', \'data\'])')
        ):
            extract._ServiceLayer.check_capabilities(mocker.Mock(layer_properties_json=response_json), 'query')

    def test_check_capabilities_passes_on_differing_cases(self, mocker):
        response_json = {'capabilities': 'Map,Query,Data'}

        implicit_return = extract._ServiceLayer.check_capabilities(
            mocker.Mock(layer_properties_json=response_json), 'query'
        )
        assert implicit_return is None

    def test_check_layer_type_allows_feature_layer(self, mocker):
        response_json = {'type': 'Feature Layer'}

        implicit_return = extract._ServiceLayer._check_layer_type(mocker.Mock(layer_properties_json=response_json))
        assert implicit_return is None

    def test_check_layer_type_allows_table(self, mocker):
        response_json = {'type': 'Table'}

        implicit_return = extract._ServiceLayer._check_layer_type(mocker.Mock(layer_properties_json=response_json))
        assert implicit_return is None

    def test_check_layer_type_raises_on_group_layer(self, mocker):
        response_json = {'type': 'Group Layer'}

        with pytest.raises(RuntimeError, match='Layer foo.bar/0 is a Group Layer, not a feature layer or table'):
            extract._ServiceLayer._check_layer_type(
                mocker.Mock(layer_properties_json=response_json, layer_url='foo.bar/0')
            )

    def test_get_object_id_field_gets_field_from_properties(self, mocker):
        response_json = {'objectIdField': 'OBJECTID_1'}

        assert extract._ServiceLayer._get_object_id_field(
            mocker.Mock(layer_properties_json=response_json)
        ) == 'OBJECTID_1'

    def test_get_object_id_field_uses_OBJECTID_if_no_field(self, mocker):
        response_json = {}

        assert extract._ServiceLayer._get_object_id_field(
            mocker.Mock(layer_properties_json=response_json)
        ) == 'OBJECTID'
