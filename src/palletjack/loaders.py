"""Classes for loading data from various sources into a pandas dataframe
"""

import logging
from pathlib import Path

import pandas as pd
import pygsheets
import pysftp


class GSheetLoader:
    """Loads data from a Google Sheets spreadsheet into a pandas data frame"""

    def __init__(self, service_file):
        self.gsheets_client = pygsheets.authorize(service_file=service_file)
        self._class_logger = logging.getLogger(__name__).getChild(self.__class__.__name__)

    def load_specific_worksheet_into_dataframe(self, sheet_id, worksheet, by_title=False):
        """Load a single worksheet from a spreadsheet into a dataframe by worksheet index or title

        Args:
            sheet_id (str): The ID of the sheet (long alpha-numeric unique ID)
            worksheet (int or str): Zero-based index of the worksheet or the worksheet title
            by_title (bool, optional): Search for worksheet by title instead of index. Defaults to False.

        Returns:
            pd.DataFrame: The specified worksheet as a data frame.
        """

        self._class_logger.debug('Loading sheet ID %s', sheet_id)
        sheet = self.gsheets_client.open_by_key(sheet_id)

        if by_title:
            self._class_logger.debug('Loading worksheet by title %s', worksheet)
            return sheet.worksheet_by_title(worksheet).get_as_df()
        else:
            self._class_logger.debug('Loading worksheet by index %s', worksheet)
            return sheet.worksheet(worksheet).get_as_df()

    def load_all_worksheets_into_dataframes(self, sheet_id):
        """Load all worksheets into a dictionary of dataframes. Keys are the worksheet.

        Args:
            sheet_id (str): The ID of the sheet (long alpha-numeric unique ID)

        Returns:
            dict: {'worksheet_name': Worksheet as a dataframe}
        """

        self._class_logger.debug('Loading sheet ID %s', sheet_id)
        sheet = self.gsheets_client.open_by_key(sheet_id)

        worksheet_dfs = {worksheet.title: worksheet.get_as_df() for worksheet in sheet.worksheets()}

        return worksheet_dfs

    @staticmethod
    def combine_worksheets_into_single_dataframe(worksheet_dfs):
        """Merge worksheet dataframes (having same columns) into a single dataframe with a new 'worksheet' column
        identifying the source worksheet.

        Args:
            worksheet_dfs (dict): {'worksheet_name': Worksheet as a dataframe}.

        Raises:
            ValueError: If all the worksheets in worksheets_dfs don't have the same column index, it raises an error
            and bombs out.

        Returns:
            pd.DataFrame: A single combined data frame with a new 'worksheet' column identifying the worksheet the row
            came from. The row index is the original row numbers and is probably not unique.
        """

        dataframes = list(worksheet_dfs.values())

        #: Make sure all the dataframes have the same columns
        if not all([set(dataframes[0].columns) == set(df.columns) for df in dataframes]):
            raise ValueError('Columns do not match; cannot create mutli-index dataframe')

        concatted_df = pd.concat(dataframes, keys=worksheet_dfs.keys(), names=['worksheet', 'row'])
        return concatted_df.reset_index(level='worksheet')


class SFTPLoader:
    """Loads data from an SFTP share into a pandas DataFrame
    """

    def __init__(self, host, username, password, knownhosts_file, download_dir):
        self.host = host
        self.username = username
        self.password = password
        self.knownhosts_file = knownhosts_file
        self.download_dir = download_dir
        self._class_logger = logging.getLogger(__name__).getChild(self.__class__.__name__)

    def download_sftp_folder_contents(self, sftp_folder='upload'):
        """Download all files in sftp_folder to the SFTPLoader's download_dir

        Args:
            sftp_folder (str, optional): Path of remote folder, relative to sftp home directory. Defaults to 'upload'.
        """

        self._class_logger.info('Downloading files from `%s:%s` to `%s`', self.host, sftp_folder, self.download_dir)
        starting_file_count = len(list(self.download_dir.iterdir()))
        self._class_logger.debug('SFTP Username: %s', self.username)
        connection_opts = pysftp.CnOpts(knownhosts=self.knownhosts_file)
        with pysftp.Connection(
            self.host, username=self.username, password=self.password, cnopts=connection_opts
        ) as sftp:
            try:
                sftp.get_d(sftp_folder, self.download_dir, preserve_mtime=True)
            except FileNotFoundError as error:
                raise FileNotFoundError(f'Folder `{sftp_folder}` not found on SFTP server') from error
        downloaded_file_count = len(list(self.download_dir.iterdir())) - starting_file_count
        if not downloaded_file_count:
            raise ValueError('No files downloaded')
        return downloaded_file_count

    def download_sftp_single_file(self, filename, sftp_folder='upload'):
        """Download filename into SFTPLoader's download_dir

        Args:
            filename (str): Filename to download; used as output filename as well.
            sftp_folder (str, optional): Path of remote folder, relative to sftp home directory. Defaults to 'upload'.

        Raises:
            FileNotFoundError: Will warn if pysftp can't find the file or folder on the sftp server

        Returns:
            Path: Downloaded file's path
        """

        outfile = Path(self.download_dir, filename)

        self._class_logger.info('Downloading %s from `%s:%s` to `%s`', filename, self.host, sftp_folder, outfile)
        self._class_logger.debug('SFTP Username: %s', self.username)
        connection_opts = pysftp.CnOpts(knownhosts=self.knownhosts_file)
        try:
            with pysftp.Connection(
                self.host,
                username=self.username,
                password=self.password,
                cnopts=connection_opts,
                default_path=sftp_folder,
            ) as sftp:
                sftp.get(filename, localpath=outfile, preserve_mtime=True)
        except FileNotFoundError as error:
            raise FileNotFoundError(f'File `{filename}` or folder `{sftp_folder}`` not found on SFTP server') from error

        return outfile

    def read_csv_into_dataframe(self, filename, column_types=None):
        """Read filename into a dataframe with optional column names and types

        Args:
            filename (str): Name of file in the SFTPLoader's download_dir
            column_types (dict, optional): Column names and their dtypes(np.float64, str, etc). Defaults to None.

        Returns:
            pd.DataFrame: CSV as a pandas dataframe
        """

        self._class_logger.info('Reading `%s` into dataframe', filename)
        filepath = Path(self.download_dir, filename)
        column_names = None
        if column_types:
            column_names = list(column_types.keys())
        dataframe = pd.read_csv(filepath, names=column_names, dtype=column_types)
        self._class_logger.debug('Dataframe shape: %s', dataframe.shape)
        if len(dataframe.index) == 0:
            self._class_logger.warning('Dataframe contains no rows. Shape: %s', dataframe.shape)
        return dataframe
