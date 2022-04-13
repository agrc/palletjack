from http import client

import pandas as pd
import pytest
from pandas import testing as tm

from palletjack import loaders


class TestGSheetsLoader:

    def test_load_specific_worksheet_into_dataframe_by_id(self, mocker):
        sheet_mock = mocker.Mock()

        client_mock = mocker.Mock()
        client_mock.open_by_key.return_value = sheet_mock

        gsheet_loader_mock = mocker.Mock()
        gsheet_loader_mock.gsheets_client = client_mock

        loaders.GSheetLoader.load_specific_worksheet_into_dataframe(gsheet_loader_mock, 'foobar', 5)

        sheet_mock.worksheet.assert_called_once_with(5)
        sheet_mock.worksheet_by_title.assert_not_called()

    def test_load_specific_worksheet_into_dataframe_by_title(self, mocker):
        sheet_mock = mocker.Mock()

        client_mock = mocker.Mock()
        client_mock.open_by_key.return_value = sheet_mock

        gsheet_loader_mock = mocker.Mock()
        gsheet_loader_mock.gsheets_client = client_mock

        loaders.GSheetLoader.load_specific_worksheet_into_dataframe(gsheet_loader_mock, 'foobar', '2015', by_title=True)

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

        df_dict = loaders.GSheetLoader.load_all_worksheets_into_dataframes(gsheet_loader_mock, 'foobar')

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

        df_dict = loaders.GSheetLoader.load_all_worksheets_into_dataframes(gsheet_loader_mock, 'foobar')

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

        combined_df = loaders.GSheetLoader.combine_worksheets_into_single_dataframe(df_dict)

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

        with pytest.raises(ValueError) as error:
            combined_df = loaders.GSheetLoader.combine_worksheets_into_single_dataframe(df_dict)

        assert 'Columns do not match; cannot create mutli-index dataframe' in str(error.value)
