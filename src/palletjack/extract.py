"""Extract tabular/spatial data from various sources into a pandas dataframe.

Each different type of source has its own class. Each class may have multiple methods available for different loading
operations or techniques.
"""

import json
import logging
import mimetypes
import random
import re
import time
import warnings
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from string import Template
from time import sleep

import arcgis
import geopandas as gpd
import pandas as pd
import pysftp
import requests
import sqlalchemy
import ujson
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

        self._class_logger.debug("Loading sheet ID %s", sheet_id)
        sheet = self.gsheets_client.open_by_key(sheet_id)

        if by_title:
            self._class_logger.debug("Loading worksheet by title %s", worksheet)
            return sheet.worksheet_by_title(worksheet).get_as_df()
        else:
            self._class_logger.debug("Loading worksheet by index %s", worksheet)
            return sheet.worksheet("index", worksheet).get_as_df()

    def load_all_worksheets_into_dataframes(self, sheet_id):
        """Load all worksheets into a dictionary of dataframes. Keys are the worksheet.

        Args:
            sheet_id (str): The ID of the sheet (long alpha-numeric unique ID)

        Returns:
            dict: {'worksheet_name': Worksheet as a dataframe}
        """

        self._class_logger.debug("Loading sheet ID %s", sheet_id)
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
            raise ValueError("Columns do not match; cannot create multi-index dataframe")

        self._class_logger.debug("Concatting worksheet dataframes %s into a single dataframe", worksheet_dfs.keys())
        concatted_df = pd.concat(dataframes, keys=worksheet_dfs.keys(), names=["worksheet", "row"])
        return concatted_df.reset_index(level="worksheet")


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
        self._class_logger.debug("Initializing GoogleDriveDownloader")
        self._class_logger.debug("Output directory: %s", out_dir)
        self.out_dir = Path(out_dir)
        regex_pattern = "(\/|=)([-\w]{25,})"  # pylint:disable=anomalous-backslash-in-string
        self._class_logger.debug("Regex pattern: %s", regex_pattern)
        self.regex = re.compile(regex_pattern)

    @staticmethod
    def _save_response_content(response, destination, chunk_size=32768):
        """Download streaming response content in chunks

        Args:
            response (requests.response): The response object from the requests .get call
            destination (Path): File path to write to
            chunk_size (int, optional): Download the file in chunks of this size. Defaults to 32768.
        """

        with destination.open(mode="wb") as out_file:
            for chunk in response.iter_content(chunk_size):
                if chunk:  # filter out keep-alive new chunks
                    out_file.write(chunk)

    def _get_http_response(self, file_id, base_url="https://docs.google.com/uc?export=download"):
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

        response = requests.get(base_url, params={"id": file_id}, stream=True)

        if "text/html" in response.headers["Content-Type"]:
            self._class_logger.error(response.headers)
            raise RuntimeError(f"Cannot access {file_id} (is it publicly shared?). Response header in log.")

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

        content = response.headers["Content-Disposition"]
        all_filenames = re.findall("filename\*?=([^;]+)", content, flags=re.IGNORECASE)  # pylint:disable=anomalous-backslash-in-string
        if all_filenames:
            #: Remove spurious whitespace and "s
            return all_filenames[0].strip().strip('"')

        #: If we don't return a filename, raise an error instead
        raise ValueError("filename not found in response header")

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
                raise IndexError(f"Regex could not extract the file id from sharing link {sharing_link}") from err
        raise RuntimeError(f"Regex could not match sharing link {sharing_link}")

    def download_file_from_google_drive(self, sharing_link, join_id, pause=0.0):
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
            self._class_logger.debug("Sleeping for %s", pause)
        sleep(pause)
        if not sharing_link:
            self._class_logger.debug("Row %s has no attachment info", join_id)
            return None
        self._class_logger.debug("Row %s: downloading shared file %s", join_id, sharing_link)
        try:
            file_id = self._get_file_id_from_sharing_link(sharing_link)
            self._class_logger.debug("Row %s: extracted file id %s", join_id, file_id)
            response = self._get_http_response(file_id)
            filename = self._get_filename_from_response(response)
            out_file_path = self.out_dir / filename
            self._class_logger.debug("Row %s: writing to %s", join_id, out_file_path)
            self._save_response_content(response, out_file_path)
            return out_file_path
        except Exception as err:
            self._class_logger.warning("Row %s: Couldn't download %s", join_id, sharing_link)
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
        filename = metadata["name"]
        if not Path(filename).suffix:
            try:
                filename = filename + mimetypes.guess_extension(metadata["mimeType"])
            except KeyError:
                self._class_logger.warning("%s: No MIME type in drive info, file extension not set", file_id)
            except TypeError:
                self._class_logger.warning(
                    "%s: Unable to determine file extension from MIME type, file extension not set", file_id
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
        """Download a file using the Google API via pygsheets authentication.

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
            self._class_logger.debug("Row %s has no attachment info", join_id)
            return None
        self._class_logger.debug("Row %s: downloading file %s", join_id, sharing_link)
        try:
            file_id = self._get_file_id_from_sharing_link(sharing_link)
            self._class_logger.debug("Row %s: extracted file id %s", join_id, file_id)
            get_media_request, filename = utils.retry(
                self._get_request_and_filename_from_drive_api, gsheets_client, file_id
            )
            out_file_path = self.out_dir / filename
            self._class_logger.debug("Row %s: writing to %s", join_id, out_file_path)
            utils.retry(self._save_get_media_content, get_media_request, out_file_path)
            return out_file_path
        except Exception as err:
            self._class_logger.warning("Row %s: Couldn't download %s", join_id, sharing_link)
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
            axis=1,
        )
        return dataframe


class SFTPLoader:
    """Loads data from an SFTP share into a pandas DataFrame"""

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

    def download_sftp_folder_contents(self, sftp_folder="upload"):
        """Download all files in sftp_folder to the SFTPLoader's download_dir

        Args:
            sftp_folder (str, optional): Path of remote folder, relative to sftp home directory. Defaults to 'upload'.
        """

        self._class_logger.info("Downloading files from `%s:%s` to `%s`", self.host, sftp_folder, self.download_dir)
        starting_file_count = len(list(self.download_dir.iterdir()))
        self._class_logger.debug("SFTP Username: %s", self.username)
        connection_opts = pysftp.CnOpts(knownhosts=self.knownhosts_file)
        with pysftp.Connection(
            self.host, username=self.username, password=self.password, cnopts=connection_opts
        ) as sftp:
            try:
                sftp.get_d(sftp_folder, self.download_dir, preserve_mtime=True)
            except FileNotFoundError as error:
                raise FileNotFoundError(f"Folder `{sftp_folder}` not found on SFTP server") from error
        downloaded_file_count = len(list(self.download_dir.iterdir())) - starting_file_count
        if not downloaded_file_count:
            raise ValueError("No files downloaded")
        return downloaded_file_count

    def download_sftp_single_file(self, filename, sftp_folder="upload"):
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

        self._class_logger.info("Downloading %s from `%s:%s` to `%s`", filename, self.host, sftp_folder, outfile)
        self._class_logger.debug("SFTP Username: %s", self.username)
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
            raise FileNotFoundError(f"File `{filename}` or folder `{sftp_folder}`` not found on SFTP server") from error

        return outfile

    def read_csv_into_dataframe(self, filename, column_types=None):
        """Read filename into a dataframe with optional column names and types

        Args:
            filename (str): Name of file in the SFTPLoader's download_dir
            column_types (dict, optional): Column names and their dtypes(np.float64, str, etc). Defaults to None.

        Returns:
            pd.DataFrame: CSV as a pandas dataframe
        """

        self._class_logger.info("Reading `%s` into dataframe", filename)
        filepath = Path(self.download_dir, filename)
        column_names = None
        if column_types:
            column_names = list(column_types.keys())
        dataframe = pd.read_csv(filepath, names=column_names, dtype=column_types)
        self._class_logger.debug("Dataframe shape: %s", dataframe.shape)
        if len(dataframe.index) == 0:
            self._class_logger.warning("Dataframe contains no rows. Shape: %s", dataframe.shape)
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
        if utils.is_running_in_gcp():
            self._class_logger.info("running in GCP, using unix socket")
            self.engine = sqlalchemy.create_engine(
                sqlalchemy.engine.url.URL.create(
                    drivername="postgresql+pg8000",
                    username=username,
                    password=password,
                    database=database,
                    query={"unix_sock": f"/cloudsql/{host}/.s.PGSQL.{port}"},  #: requires the pg8000 package
                )
            )
        else:
            self._class_logger.info("running locally, using traditional host connection")
            self.engine = sqlalchemy.create_engine(
                sqlalchemy.engine.url.URL.create(
                    drivername="postgresql",
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

        self._class_logger.info("Reading `%s` into dataframe", table_name)

        dataframe = gpd.read_postgis(
            f"select * from {table_name}", self.engine, index_col=index_column, crs=crs, geom_col=spatial_column
        )

        spatial_dataframe = pd.DataFrame.spatial.from_geodataframe(dataframe, column_name=spatial_column)

        self._class_logger.debug("Dataframe shape: %s", spatial_dataframe.shape)
        if len(spatial_dataframe.index) == 0:
            self._class_logger.warning("Dataframe contains no rows. Shape: %s", spatial_dataframe.shape)

        return spatial_dataframe


class RESTServiceLoader:
    """Downloads features from a layer within a map service or feature service (with queries enabled) based on its REST
    endpoint.

    Create a RestServiceLoader object to represent the service and call get_features on the RESTServiceLoader object,
    passing in a ServiceLayer object representing the layer you want to download. This will use either a specified
    chunk size or the service's maxRecordCount to download the data in appropriately-sized chunks using the OIDs
    returned by the service. It will retry individual chunks three times in case of error to ensure the best chance of
    success.
    """

    def __init__(self, service_url, timeout=5, token=None):
        """Create a representation of a REST FeatureService or MapService

        Args:
            service_url (str): The service's REST endpoint
            timeout (int, optional): Timeout for HTTP requests in seconds. Defaults to 5.
            token (str, optional): Auth token for the service. Defaults to None.
        """

        if service_url[-1] == "/":
            service_url = service_url[:-1]
        self.url = service_url
        self.timeout = timeout
        self.token = token

        self._class_logger = logging.getLogger(__name__).getChild(self.__class__.__name__)

    def get_features(self, service_layer, chunk_size=100):
        """Download the features from a ServiceLayer by unique ID in chunk_size-limited requests.

        Uses either chunk_size or the service's maxRecordCount parameter to chunk the request into manageable-sized
        requests. 100 seems to be the sweet spot before requests start to error out consistently. To limit the number
        of features returned, you can specify a geographic bounding box or where clause when creating the ServiceLayer.
        Individual chunk requests and other HTML requests are wrapped in retries to handle momentary network glitches.

        This method does not catch any errors raised by the individual ServiceLayer methods it calls. See that class
        for more information on the individual errors.

        Args:
            service_layer (ServiceLayer): A ServiceLayer object representing the layer to download
            chunk_size (int, optional): Number of features to download per chunk. Defaults to 100. If set to None, it
                will use the service's maxRecordCount. Adjust if the service is failing frequently.

        Returns:
            pd.DataFrame.spatial: The service's features as a spatially-enabled dataframe
        """

        self._class_logger.info("Getting features from %s...", service_layer.layer_url)
        self._class_logger.debug("Checking for query capability...")
        service_layer.check_capabilities("query")
        max_record_count = chunk_size
        if not max_record_count:
            self._class_logger.debug("Getting max record count...")
            max_record_count = service_layer.max_record_count
        self._class_logger.debug("Getting object ids...")
        oids = utils.retry(service_layer.get_object_ids)

        if len(oids) == 0:
            warnings.warn(f"Layer {service_layer.layer_url} has no features")
            return None

        self._class_logger.debug("Downloading %s features in chunks of %s...", len(oids), max_record_count)
        all_features_df = pd.DataFrame()
        for oid_subset in utils.chunker(oids, max_record_count):
            # sleep between 1.5 and 3 s to be friendly
            time.sleep(random.randint(150, 300) / 100)
            all_features_df = pd.concat(
                [
                    all_features_df,
                    utils.retry(service_layer.get_unique_id_list_as_dataframe, service_layer.oid_field, oid_subset),
                ],
                ignore_index=True,
            )
        #: if you don't do ignore_index=True, the index won't be unique and this will trip up esri's spatial dataframe
        #: which tries calling [geom_col][0] to determine the geometry type, which then returns a series instead of a
        #: single value because the index isn't unique

        return all_features_df

    def get_feature_layers_info(self):
        """Get the information dictionary for any and all feature layers and tables within the service.

        Retries the request to the service's REST endpoint three times in case of error.

        Raises:
            RuntimeError: If the response can't be parsed as JSON, the service does not contain layers, or the response
                does not contain information about the layer types

        Returns:
            dict: The parsed JSON info of the service's feature layers
        """

        response = utils.retry(self._get_service_info)

        try:
            response_json = response.json()
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Could not parse response from {self.url}") from error

        try:
            layers = [layer for layer in response_json["layers"] if layer["type"] in ["Feature Layer", "Table"]]
        except KeyError as error:
            if "layers" in str(error):
                raise RuntimeError(f"Response from {self.url} does not contain layer information") from error
            raise RuntimeError("Layer info did not contain layer type") from error

        return layers

    def _get_service_info(self):
        """Get the basic info from the service's REST endpoint

        Raises:
            requests.HTTPError: If the response returns a status code considered an error

        Returns:
            requests.Response: Raw response object from a /query request.
        """
        self._class_logger.debug("Getting service information...")

        params = {"f": "json"}

        if self.token:
            params["token"] = self.token

        response = requests.get(f"{self.url}/query", params=params, timeout=self.timeout)

        response.raise_for_status()

        return response


class ServiceLayer:
    """Represents a single layer within a service and provides methods for querying the layer and downloading features.

    The methods in this object represent the individual steps of a more complete download process involving sanity
    checks, getting the list of unique IDs to download, and then downloading based on those unique IDS. Any queries,
    where clauses, or subsetting can be done by specifying the envelope_params, feature_params, and/or where_clause to
    limit the unique IDs used to download the features. The get_features method in the RESTServiceLoader class handles
    all of these steps and should be used instead of calling the methods in this class directly.
    """

    def __init__(self, layer_url, timeout=5, envelope_params=None, feature_params=None, where_clause="1=1", token=None):
        """Create an object representing a single layer

        Args:
            layer_url (str): The full URL to the desired layer
            timeout (int, optional): Timeout for HTTP requests in seconds. Defaults to 5.
            envelope_params (dict, optional): Bounding box and it's spatial reference to spatially limit feature
                collection in the form {'geometry': '{xmin},{ymin},{xmax},{ymax}', 'inSR': '{wkid}'}. Defaults to None.
            feature_params (dict, optional): Additional query parameters to pass to the service when downloading
                features. Parameter defaults to None, and the query defaults to 'outFields': '*', 'returnGeometry':
                'true'. See the ArcGIS REST API documentation for more information.
            where_clause (str, optional): Where clause to refine the features returned. Defaults to '1=1'.
            token (str, optional): Auth token for the service. Defaults to None.

        """

        self._class_logger = logging.getLogger(__name__).getChild(self.__class__.__name__)

        if layer_url[-1] == "/":
            layer_url = layer_url[:-1]
        self.layer_url = layer_url
        self.timeout = timeout
        self.token = token

        self.layer_properties_json = self._get_layer_info()
        self.max_record_count = self.layer_properties_json["maxRecordCount"]
        self.oid_field = self._get_object_id_field()
        self._check_layer_type()

        self.envelope_params = None
        if envelope_params:
            try:
                envelope_params["geometry"] and envelope_params["inSR"]
            except KeyError as error:
                raise ValueError(
                    "envelope_params must contain both the envelope geometry and its spatial reference"
                ) from error
            self.envelope_params = {"geometryType": "esriGeometryEnvelope"}
            self.envelope_params.update(envelope_params)

        self.feature_params = {"outFields": "*", "returnGeometry": "true"}
        if feature_params:
            self.feature_params.update(feature_params)

        self.where_clause = where_clause

    def _get_layer_info(self):
        """Do a basic query to get the layer's information as a dictionary from the json response.

        Raises:
            RuntimeError: If the response does not contain 'capabilities', 'type', or 'maxRecordCount' keys.

        Returns:
            dict: The query's json response as a dictionary
        """
        params = {"f": "json"}
        if self.token:
            params["token"] = self.token
        response_json = utils.retry(requests.get, self.layer_url, params=params, timeout=self.timeout).json()
        try:
            #: bogus boolean to make sure the keys exist
            response_json["capabilities"] and response_json["type"]  # and response_json['maxRecordCount']
        except KeyError as error:
            raise RuntimeError(
                "Response does not contain layer information; ensure URL points to a valid layer"
            ) from error

        try:
            response_json["maxRecordCount"]
        except KeyError as error:
            raise RuntimeError(
                "Response does not contain maxRecordCount; ensure URL points to a valid layer and is not a Group Layer"
            ) from error

        return response_json

    def check_capabilities(self, capability):
        """Raise error if the layer does not support capability

        Args:
            capability (str): The capability in question; will be casefolded.

        Raises:
            RuntimeError: if the casefolded capability is not present in the layer's capabilities.
        """
        layer_capabilities = self.layer_properties_json["capabilities"].casefold().split(",")
        if capability.casefold() not in layer_capabilities:
            raise RuntimeError(f"{capability.casefold()} capability not in layer's capabilities ({layer_capabilities})")

    def _check_layer_type(self):
        """Make sure the layer is a feature layer or table (and thus we can extract features from it)

        Args:
            response_json (dict): The JSON response from a basic query parsed as a dictionary.

        Raises:
            RuntimeError: If the REST response type is not Feature Layer or Table
        """

        if self.layer_properties_json["type"] not in ["Feature Layer", "Table"]:
            raise RuntimeError(
                f"Layer {self.layer_url} is a {self.layer_properties_json['type']}, not a feature layer or table"
            )

    def _get_object_id_field(self):
        """Get the service's objectIdField attribute

        Args:
            response_json (dict): The JSON response from a basic query parsed as a dictionary.

        Returns:
            str: objectIdField
        """

        try:
            unique_id_field = self.layer_properties_json["objectIdField"]
        except KeyError:
            self._class_logger.debug("No objectIdField found in %s, using OBJECTID instead", self.layer_url)
            unique_id_field = "OBJECTID"

        return unique_id_field

    def get_object_ids(self):
        """Get the Object IDs of the feature service layer, using the bounding envelope and/or where clause if present.

        Raises:
            RuntimeError: If the response does not contain an 'objectIds' key.

        Returns:
            list(int): The Object IDs
        """

        objectid_params = {"returnIdsOnly": "true", "f": "json", "where": self.where_clause}
        if self.envelope_params is not None:
            objectid_params.update(self.envelope_params)
        self._class_logger.debug("OID params: %s", objectid_params)
        if self.token:
            objectid_params["token"] = self.token

        response = utils.retry(requests.get, f"{self.layer_url}/query", params=objectid_params, timeout=self.timeout)
        oids = []
        try:
            oids = sorted(response.json()["objectIds"])
        except KeyError as error:
            raise RuntimeError(f"Could not get object IDs from {self.layer_url}") from error
        except TypeError as error:
            if "'NoneType' object is not iterable" in str(error):
                oids = []

        return oids

    def get_unique_id_list_as_dataframe(self, unique_id_field, unique_id_list):
        """Use a REST query to download specified features from a MapService or FeatureService layer.

        unique_id_list defines the ids in unique_id_field to download.

        Args:
            unique_id_field (str): The field in the service layer used as the unique ID.
            unique_id_list (list): The list of unique IDs to download.

        Raises:
            Runtime Error: If the chunk's HTTP response code is not 200 (ie, the request failed)
            Runtime Error: The response could not be parsed into JSON (a technically successful request but bad data)
            Runtime Error: If the number of features downloaded does not match the number of OIDs requested

        Returns:
            pd.DataFrame.spatial: Spatially-enabled dataframe of the service layer's features
        """
        unique_id_params = {"f": "json"}
        unique_id_params.update(self.feature_params)
        unique_id_params.update({"where": f"{unique_id_field} in ({','.join([str(oid) for oid in unique_id_list])})"})

        if self.token:
            unique_id_params["token"] = self.token

        self._class_logger.debug("OID range params: %s", unique_id_params)
        response = requests.post(f"{self.layer_url}/query", data=unique_id_params, timeout=self.timeout)

        if response.status_code != 200:
            raise RuntimeError(f"Bad chunk response HTTP status code ({response.status_code})")

        try:
            fs = arcgis.features.FeatureSet.from_json(response.text)
            features_df = fs.sdf.sort_values(by=unique_id_field)

        except ujson.JSONDecodeError as error:
            raise RuntimeError("Could not parse chunk features from response") from error

        if len(features_df) != len(unique_id_list):
            raise RuntimeError(
                f"Missing features. {len(unique_id_list)} OIDs requested, but {len(features_df)} features downloaded"
            )

        return features_df


class SalesforceApiUserCredentials:
    """A salesforce API user credential model"""

    def __init__(self, secret, key) -> None:
        self.client_secret = secret
        self.client_id = key


class SalesforceSandboxCredentials:
    """A salesforce sandbox credential model"""

    def __init__(self, username, password, token, secret, key) -> None:
        self.username = username
        self.password = password + token
        self.client_secret = secret
        self.client_id = key


class SalesforceRestLoader:
    """Queries a Salesforce organization using SOQL using the REST API.

    To use this loader a connected app needs to be created in Salesforce which create the credentials.
    You can then use the workbench, https://workbench.developerforce.com/query.php, to construct and test SOQL queries.

    Create a Salesforce credential model and a SalesforceRest loader to authenticate and query Salesforce.
    Call get_records with a SOQL query to get the results as a pandas dataframe."""

    sandbox = False

    access_token_template = Template("https://$org.my.salesforce.com/services/oauth2/token")
    access_token_url = ""
    access_token = {}

    org_template = Template("https://$org.my.salesforce.com")
    org_url = ""

    client_secret = None
    client_id = None
    username = None
    password = None

    token_lease_in_days = 30
    access_token_timeout_in_seconds = 10
    soql_query_timeout_in_seconds = 30

    def __init__(self, org, credentials, sandbox=False) -> None:
        """Create a SalesforceRestLoader to query Salesforce. Timeout values are set to default values and can be
        modified by updating the instance variables.

        token_lease_in_days: The number of days before the token expires. Defaults to 30 days.
        access_token_timeout_in_seconds: The timeout for getting a new token. Defaults to 10 seconds.
        soql_query_timeout_in_seconds: The timeout for a SOQL query. Defaults to 30 seconds.

        Args:
            org (string): the organization name from the salesforce url https://[org].my.salesforce.com
            credentials (SalesforceApiUserCredentials | SalesforceSandboxCredentials): The credentials to use to
            authenticate.
            sandbox (bool, optional): The credentials for sandboxes are different than API users. Defaults to False if
            it's not a sandbox instance of Salesforce.
        """
        self.client_secret = credentials.client_secret
        self.client_id = credentials.client_id

        if sandbox:
            self.username = credentials.username
            self.password = credentials.password
            org = org + ".sandbox"

        self.access_token_url = self.access_token_template.substitute(org=org)
        self.org_url = self.org_template.substitute(org=org)

        self.sandbox = sandbox

        self._class_logger = logging.getLogger(__name__).getChild(self.__class__.__name__)

    def _is_token_valid(self, token: dict[str, str]) -> bool:
        """Checks if the token is valid by looking at the issued_at field.

        Args:
            token (dict[str, str]): the sales force token response to check

        Returns:
            bool: true if the token is valid
        """
        if "issued_at" not in token.keys():
            return False

        issued = timedelta.max
        lease = timedelta(days=self.token_lease_in_days)

        try:
            ticks = int(token["issued_at"])
            issued = datetime.fromtimestamp(ticks / 1000)
            days_from_today = (datetime.now() - issued).days

            self._class_logger.debug("Token is %s days old", days_from_today)
        except ValueError:
            self._class_logger.warning("could not convert issued_at delta to a number %s", token)

            return False

        return datetime.now() < (issued + lease)

    def _get_token(self) -> dict[str, str]:
        """Gets a new Salesforce access token if the current one is expired."""
        if self._is_token_valid(self.access_token):
            return self.access_token

        form_data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        if self.sandbox:
            form_data["grant_type"] = "password"
            form_data["username"] = self.username
            form_data["password"] = self.password

        response = requests.post(
            self.access_token_url,
            data=form_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=self.access_token_timeout_in_seconds,
        )

        self._class_logger.debug("requesting new token %s", response)

        response.raise_for_status()

        self.access_token = response.json()

        return self.access_token

    def get_records(self, query_endpoint, query_string) -> pd.DataFrame:
        """Queries the Salesforce API and returns the results as a dataframe.

        Args:
            query_endpoint (str): The REST endpoint to use for the query. Usually "/services/data/vXX.X/query".
            query_string (str): A SOQL query string.

        Raises:
            ValueError: If the query fails.

        Returns:
            pd.Dataframe: A dataframe of the results.
        """

        response_data = self._query_records(query_endpoint, {"q": query_string})

        df = pd.DataFrame(response_data["records"])

        while response_data["done"] is False:
            response_data = self._query_records(response_data["nextRecordsUrl"])
            df = pd.concat([df, pd.DataFrame(response_data["records"])])

        return df.reset_index(drop=True)  #: the concatted index is just a repeat of 0:2000

    def _query_records(self, query_endpoint, query_params=None) -> dict:
        """Queries the Salesforce API and returns the results as a dictionary.

        Args:
            query_endpoint (str): The REST endpoint to use for the query, either 'query' or the nextRecordsUrl.
            query_params (dict): A dictionary of params holding the SOQL query. Defaults to None for paged queries.

        Raises:
            ValueError: If the query fails.

        Returns:
            dict: A dictionary of the results.
        """
        token = self._get_token()

        response = utils.retry(
            requests.get,
            f"{self.org_url}/{query_endpoint}",
            params=query_params,
            headers={
                "Authorization": f"Bearer {token['access_token']}",
            },
            timeout=self.soql_query_timeout_in_seconds,
        )

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as error:
            self._class_logger.error("Error getting records from Salesforce %s", error)

            raise ValueError(f"Error getting records from Salesforce {response.json()}") from error

        return response.json()
