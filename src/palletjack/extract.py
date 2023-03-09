"""Extract tabular/spatial data from various sources into a pandas dataframe.

Each different type of source has its own class. Each class may have multiple methods available for different loading
operations or techniques.
"""

import logging
import mimetypes
import os
import re
from io import BytesIO
from pathlib import Path
from time import sleep

import geopandas as gpd
import pandas as pd
import pysftp
import requests
import sqlalchemy
from googleapiclient.http import MediaIoBaseDownload

from palletjack import utils

logger = logging.getLogger(__name__)


class GSheetLoader:
    """Loads data from a Google Sheets spreadsheet into a pandas data frame

    Requires either the path to a service account .json file that has access to the sheet in question or a
    `google.auth.credentials.Credentials` object. Calling `google.auth.default()` in a Google Cloud Function will give
    you a tuple of a `Credentials` object and the project id. You can use this `Credentials` object to authorize
    pygsheets as the same account the Cloud Function is running under.
    """

    def __init__(self, credentials):
        """
        Args:
            credentials (str or google.auth.credentials.Credentials): Path to the service file OR credentials object
                obtained from google.auth.default() within a cloud function.
        """
        self._class_logger = logging.getLogger(__name__).getChild(self.__class__.__name__)
        self.gsheets_client = utils.authorize_pygsheets(credentials)

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
            raise ValueError('Columns do not match; cannot create multi-index dataframe')

        self._class_logger.debug('Concatting worksheet dataframes %s into a single dataframe', worksheet_dfs.keys())
        concatted_df = pd.concat(dataframes, keys=worksheet_dfs.keys(), names=['worksheet', 'row'])
        return concatted_df.reset_index(level='worksheet')


class GoogleDriveDownloader:
    """Provides methods to download any non-html file (ie, Content-Type != text/html) Google Drive file from it's
    sharing link (of the form `https://drive.google.com/file/d/big_long_id/etc`). The files may be publicly shared or shared with a service account.

    This class has two similar sets of methods. The `*_using_api` methods authenticate to the Google API using either a
    service account file or a `google.auth.credentials.Credentials` object and downloads using the API. The
    `*_using_api` methods are the most robust and should be used whenever possible.
    """

    def __init__(self, out_dir):
        """
        Args:
            out_dir (str or Path): Directory to save downloaded files. Can be reassigned later to change the directory.
        """
        self._class_logger = logging.getLogger(__name__).getChild(self.__class__.__name__)
        self._class_logger.debug('Initializing GoogleDriveDownloader')
        self._class_logger.debug('Output directory: %s', out_dir)
        self.out_dir = Path(out_dir)
        regex_pattern = '(\/|=)([-\w]{25,})'  # pylint:disable=anomalous-backslash-in-string
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

        response = requests.get(base_url, params={'id': file_id}, stream=True)

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
                return match.group(2)
            #: Not sure how this would happen (can't even figure out a test), but leaving in for safety.
            except IndexError as err:
                raise IndexError(f'Regex could not extract the file id from sharing link {sharing_link}') from err
        raise RuntimeError(f'Regex could not match sharing link {sharing_link}')

    def download_file_from_google_drive(self, sharing_link, join_id, pause=0.):
        """Download a publicly-shared image from Google Drive using it's sharing link

        Uses an anonymous HTTP request with support for sleeping between downloads to try to get around Google's
        blocking (I haven't found a good value yet).

        Logs a warning if the URL doesn't match the proper pattern or it can't extract the unique id from the sharing
        URL. Will also log a warning if the header's Content-Type is text/html, which usually indicates the HTTP
        response was an error message instead of the file.

        Args:
            sharing_link (str): The publicly-shared link to the image.
            join_id (str or int): The unique key for the row (used for reporting)
            pause (flt, optional): Pause the specified number of seconds before downloading. Defaults to 0.

        Returns:
            Path: Path of downloaded file or None if download fails/is not possible
        """

        if pause:
            self._class_logger.debug('Sleeping for %s', pause)
        sleep(pause)
        if not sharing_link:
            self._class_logger.debug('Row %s has no attachment info', join_id)
            return None
        self._class_logger.debug('Row %s: downloading shared file %s', join_id, sharing_link)
        try:
            file_id = self._get_file_id_from_sharing_link(sharing_link)
            self._class_logger.debug('Row %s: extracted file id %s', join_id, file_id)
            response = self._get_http_response(file_id)
            filename = self._get_filename_from_response(response)
            out_file_path = self.out_dir / filename
            self._class_logger.debug('Row %s: writing to %s', join_id, out_file_path)
            self._save_response_content(response, out_file_path)
            return out_file_path
        except Exception as err:
            self._class_logger.warning('Row %s: Couldn\'t download %s', join_id, sharing_link)
            self._class_logger.warning(err)
            return None

    def _get_request_and_filename_from_drive_api(self, client, file_id):
        """Get the request object and filename from a Google drive file_id. Will try to determine extension if missing.

        Args:
            client (pygsheets.Client): Authenticated client object from pygsheets
            file_id (str): The Google fileId to be downloaded

        Returns:
            (googleapiclient.http.HttpRequest, str): Result of the .get_media() call, name of the file
        """

        get_media_request = client.drive.service.files().get_media(fileId=file_id)  # pylint:disable=no-member
        metadata = client.drive.service.files().get(fileId=file_id).execute()  # pylint:disable=no-member
        filename = metadata['name']
        if not Path(filename).suffix:
            try:
                filename = filename + mimetypes.guess_extension(metadata['mimeType'])
            except KeyError:
                self._class_logger.warning('%s: No MIME type in drive info, file extension not set', file_id)
            except TypeError:
                self._class_logger.warning(
                    '%s: Unable to determine file extension from MIME type, file extension not set', file_id
                )

        return get_media_request, filename

    def _save_get_media_content(self, request, out_file_path):
        """Save the binary data referenced from a .get_media() call to out_file_path

        Args:
            request (googleapiclient.http.HttpRequest): Result of the .get_media() call
            out_file_path (Path): Path object to the location to save the data
        """

        in_memory = BytesIO()
        downloader = MediaIoBaseDownload(in_memory, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        out_file_path.write_bytes(in_memory.getbuffer())

    def download_file_from_google_drive_using_api(self, gsheets_client, sharing_link, join_id):
        """ Download a file using the Google API via pygsheets authentication.

        Requires a pygsheets client object that handles authentication.

        Logs a warning if the URL doesn't match the proper pattern or it can't extract the unique id from the sharing
        URL. Will also log a warning if the header's Content-Type is text/html, which usually indicates the HTTP
        response was an error message instead of the file.

        Args:
            gsheets_client (pygsheets.Client): The authenticated client object from pygsheets
            sharing_link (str): Sharing link to the file to be downloaded
            join_id (str or int): Unique key for the row (used for reporting)

        Returns:
            Path: Path of downloaded file or None if download fails/is not possible
        """
        if not sharing_link:
            self._class_logger.debug('Row %s has no attachment info', join_id)
            return None
        self._class_logger.debug('Row %s: downloading file %s', join_id, sharing_link)
        try:
            file_id = self._get_file_id_from_sharing_link(sharing_link)
            self._class_logger.debug('Row %s: extracted file id %s', join_id, file_id)
            get_media_request, filename = utils.retry(
                self._get_request_and_filename_from_drive_api, gsheets_client, file_id
            )
            out_file_path = self.out_dir / filename
            self._class_logger.debug('Row %s: writing to %s', join_id, out_file_path)
            utils.retry(self._save_get_media_content, get_media_request, out_file_path)
            return out_file_path
        except Exception as err:
            self._class_logger.warning('Row %s: Couldn\'t download %s', join_id, sharing_link)
            self._class_logger.warning(err)
            return None

    def download_attachments_from_dataframe(self, dataframe, sharing_link_column, join_id_column, output_path_column):
        """Download the attachments linked in a dataframe column, creating a new column with the resulting path

        Args:
            dataframe (pd.DataFrame): Input dataframe with required columns
            sharing_link_column (str): Column holding the Google sharing link
            join_id_column (str): Column holding a unique key (for reporting purposes)
            output_path_column (str): Column for the resulting path; will be added if it doesn't exist in the
                dataframe

        Returns:
            pd.DataFrame: Input dataframe with output path info
        """

        dataframe[output_path_column] = dataframe.apply(
            lambda x: self.download_file_from_google_drive(x[sharing_link_column], x[join_id_column], pause=5), axis=1
        )
        return dataframe

    def download_attachments_from_dataframe_using_api(
        self, credentials, dataframe, sharing_link_column, join_id_column, output_path_column
    ):
        """Download the attachments linked in a dataframe column using an authenticated api client, creating a new
        column with the resulting path

        Args:
            credentials (str or google.auth.credentials.Credentials): Path to the service file OR credentials object
                obtained from google.auth.default() within a cloud function.
            dataframe (pd.DataFrame): Input dataframe with required columns
            sharing_link_column (str): Column holding the Google sharing link
            join_id_column (str): Column holding a unique key (for reporting purposes)
            output_path_column (str): Column for the resulting path; will be added if it doesn't existing in the
                dataframe

        Returns:
            pd.DataFrame: Input dataframe with output path info
        """

        client = utils.authorize_pygsheets(credentials)

        dataframe[output_path_column] = dataframe.apply(
            lambda x: self.download_file_from_google_drive_using_api(client, x[sharing_link_column], x[join_id_column]),
            axis=1
        )
        return dataframe


class SFTPLoader:
    """Loads data from an SFTP share into a pandas DataFrame
    """

    def __init__(self, host, username, password, knownhosts_file, download_dir):
        """
        Args:
            host (str): The SFTP host to connect to
            username (str): SFTP username
            password (str): SFTP password
            knownhosts_file (str): Path to a known_hosts file for pysftp.CnOpts. Can be generated via ssh-keyscan.
            download_dir (str or Path): Directory to save downloaded files
        """

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


class PostgresLoader:
    """Loads data from a Postgres/PostGIS database into a pandas data frame"""

    def __init__(self, host, database, username, password, port=5432):
        """
        Args:
            host (str): Postgres server host name
            database (str): Database to connect to
            username (str): Database user
            password (str): Database password
            port (int, optional): Database port. Defaults to 5432.
        """

        self._class_logger = logging.getLogger(__name__).getChild(self.__class__.__name__)
        if os.environ.get('FUNCTION_TARGET') is not None:  #: this is an env var specific to cloud functions
            self._class_logger.info('running in GCF, using unix socket')
            self.engine = sqlalchemy.create_engine(
                sqlalchemy.engine.url.URL.create(
                    drivername='postgresql+pg8000',
                    username=username,
                    password=password,
                    database=database,
                    query={'unix_sock': f'/cloudsql/{host}/.s.PGSQL.{port}'},  #: requires the pg8000 package
                )
            )
        else:
            self._class_logger.info('running locally, using traditional host connection')
            self.engine = sqlalchemy.create_engine(
                sqlalchemy.engine.url.URL.create(
                    drivername='postgresql',
                    username=username,
                    password=password,
                    database=database,
                    host=host,
                    port=port,
                )
            )

    def read_table_into_dataframe(self, table_name, index_column, crs, spatial_column):
        """Read a table into a dataframe

        Args:
            table_name (str): Name of table or view to read in the following format: schema.table_name
            index_column (str): Name of column to use as the dataframe's index
            crs (str): Coordinate reference system of the table's geometry column
            spatial_column (str): Name of the table's geometry or geography column

        Returns:
            pd.DataFrame.spatial: Table as a spatially enabled dataframe
        """

        self._class_logger.info('Reading `%s` into dataframe', table_name)

        dataframe = gpd.read_postgis(
            f'select * from {table_name}', self.engine, index_col=index_column, crs=crs, geom_col=spatial_column
        )

        spatial_dataframe = pd.DataFrame.spatial.from_geodataframe(dataframe, column_name=spatial_column)
        for column in spatial_dataframe.select_dtypes(include=['datetime64[ns, UTC]']):
            self._class_logger.debug('Converting column `%s` to ISO string format', column)
            spatial_dataframe[column] = spatial_dataframe[column].apply(pd.Timestamp.isoformat)

        self._class_logger.debug('Dataframe shape: %s', spatial_dataframe.shape)
        if len(spatial_dataframe.index) == 0:
            self._class_logger.warning('Dataframe contains no rows. Shape: %s', spatial_dataframe.shape)

        return spatial_dataframe
