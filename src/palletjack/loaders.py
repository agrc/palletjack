import logging

import pandas as pd
import pygsheets


class GSheetLoader:

    def __init__(self, service_file):
        self.gsheets_client = pygsheets.authorize(service_file=service_file)
        self._class_logger = logging.getLogger(__name__).getChild(self.__class__.__name__)

    def load_specific_worksheet_into_dataframe(self, sheet_id, worksheet, by_title=False):

        self._class_logger.debug('Loading sheet ID %s', sheet_id)
        sheet = self.gsheets_client.open_by_key(sheet_id)

        if by_title:
            self._class_logger.debug('Loading worksheet by title %s', worksheet)
            return sheet.worksheet_by_title(worksheet).get_as_df()
        else:
            self._class_logger.debug('Loading worksheet by index %s', worksheet)
            return sheet.worksheet(worksheet).get_as_df()

    def load_all_worksheets_into_dataframes(self, sheet_id):
        self._class_logger.debug('Loading sheet ID %s', sheet_id)
        sheet = self.gsheets_client.open_by_key(sheet_id)

        worksheet_dfs = {worksheet.title: worksheet.get_as_df() for worksheet in sheet.worksheets()}

        return worksheet_dfs

    @staticmethod
    def combine_worksheets_into_single_dataframe(worksheet_dfs):
        dataframes = list(worksheet_dfs.values())

        #: Make sure all the dataframes have the same columns
        if not all([set(dataframes[0].columns) == set(df.columns) for df in dataframes]):
            raise ValueError('Columns do not match; cannot create mutli-index dataframe')

        concatted_df = pd.concat(dataframes, keys=worksheet_dfs.keys(), names=['worksheet', 'row'])
        return concatted_df.reset_index(level='worksheet')
