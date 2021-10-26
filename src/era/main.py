#!/usr/bin/env python
# * coding: utf8 *
"""
Updates the DWS ERAP layer based on their weekly
"""

from datetime import datetime

import arcgis
from supervisor.message_handlers import SendGridHandler
from supervisor.models import MessageDetails, Supervisor

from . import secrets
from .data_coupler import ColorRampReclassifier, FeatureServiceInLineUpdater, SFTPLoader


def _make_download_dir(exist_ok=False):
    today = datetime.today()
    download_dir = secrets.ERAP_BASE_DIR / today.strftime('%Y%m%d_')
    try:
        download_dir.mkdir(exist_ok=exist_ok)
    except FileNotFoundError as error:
        raise FileNotFoundError(f'Base directory {secrets.ERAP_BASE_DIR} does not exist.') from error
    else:
        return download_dir


def process():
    erap_supervisor = Supervisor()
    erap_supervisor.add_message_handler(SendGridHandler(secrets.SENDGRID_SETTINGS, 'ERA'))

    gis = arcgis.gis.GIS(secrets.AGOL_ORG, secrets.AGOL_USER, secrets.AGOL_PASSWORD)
    erap_webmap_item = gis.content.get(secrets.ERAP_WEBMAP_ITEMID)
    erap_download_dir = _make_download_dir()

    #: Load the latest data from FTP
    erap_loader = SFTPLoader(secrets, erap_download_dir)
    erap_loader.download_sftp_files()
    dataframe = erap_loader.read_csv_into_dataframe('ERAP_PAYMENTS.csv', secrets.ERAP_DATA_TYPES)

    #: Update the AGOL data
    erap_updater = FeatureServiceInLineUpdater(dataframe, 'ZipCode')
    erap_updater.update_feature_service(secrets.ERAP_FEATURE_SERVICE_URL, ['Count', 'Amount', 'LastUpdated'])

    #: Reclassify the break values on the webmap's color ramp
    erap_reclassifier = ColorRampReclassifier(erap_webmap_item, gis)
    success = erap_reclassifier.update_color_ramp_values(secrets.ERAP_LAYER_NAME, 'Amount')


if __name__ == '__main__':
    #: the code that executes if you run the file or module directly
    process()
