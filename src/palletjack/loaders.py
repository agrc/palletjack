"""Classes for loading data from various sources into a pandas dataframe
"""

import logging
import re
from pathlib import Path

import pandas as pd
import pygsheets
import pysftp
import requests

logger = logging.getLogger(__name__)


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
            return sheet.worksheet('index', worksheet).get_as_df()

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

    def combine_worksheets_into_single_dataframe(self, worksheet_dfs):
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

        self._class_logger.debug('Concatting worksheet dataframes %s into a single dataframe', worksheet_dfs.keys())
        concatted_df = pd.concat(dataframes, keys=worksheet_dfs.keys(), names=['worksheet', 'row'])
        return concatted_df.reset_index(level='worksheet')


class GoogleDriveDownloader:
    """Downloads images from publicly-shared Google Drive links (https://drive.google.com/file/d/big_long_id/etc)
    """

    def __init__(self, out_dir):
        self._class_logger = logging.getLogger(__name__).getChild(self.__class__.__name__)
        self._class_logger.debug('Initializing GoogleDriveDownloader')
        self._class_logger.debug('Output directory: %s', out_dir)
        self.out_dir = Path(out_dir)
        regex_pattern = 'https:\/\/drive.google.com\/file\/d\/(\S*)\/.*'  # pylint:disable=anomalous-backslash-in-string
        self._class_logger.debug('Regex pattern: %s', regex_pattern)
        self.regex = re.compile(regex_pattern)

    @staticmethod
    def _save_response_content(response, destination, chunk_size=32768):
        """Download streaming response content in chunks

        Args:
            response (requests.response): The response object from the requests .get call
            destination (Path): File path to write to
            chunk_size (int, optional): Download the file in chunks of this size. Defaults to 32768.
        """

        with destination.open(mode='wb') as out_file:
            for chunk in response.iter_content(chunk_size):
                if chunk:  # filter out keep-alive new chunks
                    out_file.write(chunk)

    def _get_http_response(self, file_id, base_url='https://docs.google.com/uc?export=download'):
        """Performs the HTTP GET request and checks the response

        Args:
            file_id (str): The Google-created unique id for the file
            base_url (str, optional): The base URL for the GET call. Defaults to 'https://docs.google.com/uc?
                                      export=download'.

        Raises:
            RuntimeError: If Content-Type is text/html, we can't get the file, either because it doesn't exist or isn't
                          publicly shared.

        Returns:
            request.response: The requests response object.
        """

        session = requests.Session()
        response = session.get(base_url, params={'id': file_id}, stream=True)

        if 'text/html' in response.headers['Content-Type']:
            self._class_logger.error(response.headers)
            raise RuntimeError(f'Cannot access {file_id} (is it publicly shared?). Response header in log.')

        return response

    @staticmethod
    def _get_filename_from_response(response):
        """Get the filename from the response header

        Args:
            response (requests.response): response object from the HTTP GET call

        Raises:
            ValueError: If it can't find the filename in the header

        Returns:
            str: Filename as defined in the header
        """

        content = response.headers['Content-Disposition']
        all_filenames = re.findall('filename\*?=([^;]+)', content, flags=re.IGNORECASE)  # pylint:disable=anomalous-backslash-in-string
        if all_filenames:
            #: Remove spurious whitespace and "s
            return all_filenames[0].strip().strip('"')

        #: If we don't return a filename, raise an error instead
        raise ValueError('filename not found in response header')

    def _get_file_id_from_sharing_link(self, sharing_link):
        """Use regex to parse out the unique Google id from the sharing link

        Args:
            sharing_link (str): The public sharing link to the file

        Raises:
            IndexError: If the regex matches the url but can't get a sharing roup (may not ever occur)
            RuntimeError: If the regex doesn't match the sharing link

        Returns:
            str: The unique Google id for the file.
        """

        match = self.regex.search(sharing_link)
        if match:
            try:
                return match.group(1)
            #: Not sure how this would happen (can't even figure out a test), but leaving in for safety.
            except IndexError as err:
                raise IndexError(f'Regex could not extract the file id from sharing link {sharing_link}') from err
        raise RuntimeError(f'Regex could not match sharing link {sharing_link}')

    def download_image_from_google_drive(self, sharing_link):
        """Download a publicly-shared image from Google Drive using it's sharing link

        Args:
            sharing_link (str): The publicly-shared link to the image.
        """

        file_id = self._get_file_id_from_sharing_link(sharing_link)
        self._class_logger.debug('Downloading file id %s', file_id)
        response = self._get_http_response(file_id)
        filename = self._get_filename_from_response(response)
        self._save_response_content(response, self.out_dir / filename)


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
