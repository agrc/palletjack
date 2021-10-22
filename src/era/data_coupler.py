"""Objects for bringing data from SFTP-hosted CSVs into AGOL Feature Services
"""

from pathlib import Path

import arcpy
import pandas as pd
import pysftp


class SFTPLoader:
    """Loads data from an SFTP share into a pandas DataFrame
    """

    def __init__(self, secrets, download_dir):
        self.secrets = secrets
        self.download_dir = download_dir

    def download_sftp_files(self, sftp_folder='upload'):
        """Download all files in sftp_folder to the SFTPLoader's download_dir

        Args:
            sftp_folder (str, optional): Path of remote folder, relative to sftp home directory. Defaults to 'upload'.
        """

        connection_opts = pysftp.CnOpts(knownhosts=self.secrets.KNOWNHOSTS)
        with pysftp.Connection(
            self.secrets.SFTP_HOST,
            username=self.secrets.SFTP_USERNAME,
            password=self.secrets.SFTP_PASSWORD,
            cnopts=connection_opts
        ) as sftp:
            sftp.get_d(sftp_folder, self.download_dir, preserve_mtime=True)

    def read_csv_into_dataframe(self, filename, column_types=None):
        """Read filename into a dataframe with optional column names and types

        Args:
            filename (str): Name of file in the SFTPLoader's download_dir
            column_types (dict, optional): Column names and their dtypes(np.float64, str, etc). Defaults to None.

        Returns:
            pd.DataFrame: CSV as a pandas dataframe
        """

        filepath = Path(self.download_dir, filename)
        column_names = None
        if column_types:
            column_names = column_types.keys()
        dataframe = pd.read_csv(filepath, names=column_names, dtype=column_types)
        return dataframe


class FeatureServiceInLineUpdater:
    """Updates an AGOL Feature Service with data from a pandas DataFrame
    """

    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.data_as_dict = {}

    def dataframe_to_dict(self, index_column):
        self.data_as_dict = self.dataframe.set_index(index_column).to_dict('index')

    def update_feature_service(self, feature_service_url, fields):
        with arcpy.da.UpdateCursor(feature_service_url, fields) as update_cursor:
            for row in update_cursor:
                if row[0] in self.data_as_dict:
                    #: TODO: update the row based on the values in self.data_as_dict[row[0]]
                    pass


class FeatureServiceOverwriter:
    """Overwrites an AGOL Feature Service with data from a pandas DataFrame and a geometry source (Spatially-enabled
    Data Frame, feature class, etc)
    """
    raise NotImplementedError
