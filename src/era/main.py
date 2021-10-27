#!/usr/bin/env python
# * coding: utf8 *
"""
Updates the DWS ERAP layer based on their weekly
"""

import logging
import sys
from datetime import datetime

import arcgis
from supervisor.message_handlers import SendGridHandler
from supervisor.models import MessageDetails, Supervisor

from . import secrets
from .data_coupler import ColorRampReclassifier, FeatureServiceInLineUpdater, SFTPLoader

#: Either set this logger's name to 'era' (if we use __name__ here it becomes era.main, and any other modules won't be
#:  it's children) or define it in __init__.py
erap_logger = logging.getLogger('era')
erap_logger.setLevel(logging.DEBUG)
cli_handler = logging.StreamHandler(sys.stdout)
cli_handler.setLevel(logging.DEBUG)
cli_formatter = logging.Formatter(
    fmt='%(levelname)-7s %(asctime)s %(module)10s:%(lineno)5s %(message)s', datefmt='%m-%d %H:%M:%S'
)
cli_handler.setFormatter(cli_formatter)
erap_logger.addHandler(cli_handler)


def _make_download_dir(exist_ok=False):
    today = datetime.today()
    download_dir = secrets.ERAP_BASE_DIR / today.strftime('%Y%m%d_%H%M%S')
    try:
        download_dir.mkdir(exist_ok=exist_ok)
    except FileNotFoundError as error:
        raise FileNotFoundError(f'Base directory {secrets.ERAP_BASE_DIR} does not exist.') from error
    else:
        return download_dir


def process():
    erap_logger.debug('Creating Supervisor object')
    erap_supervisor = Supervisor(logger=erap_logger)
    erap_supervisor.add_message_handler(
        SendGridHandler(sendgrid_settings=secrets.SENDGRID_SETTINGS, project_name='era')
    )

    erap_logger.debug('Logging into `%s` as `%s`', secrets.AGOL_ORG, secrets.AGOL_USER)
    gis = arcgis.gis.GIS(secrets.AGOL_ORG, secrets.AGOL_USER, secrets.AGOL_PASSWORD)
    erap_webmap_item = gis.content.get(secrets.ERAP_WEBMAP_ITEMID)
    erap_download_dir = _make_download_dir()

    #: Load the latest data from FTP
    erap_logger.info('Getting data from FTP')
    erap_loader = SFTPLoader(secrets, erap_download_dir)
    files_downloaded = erap_loader.download_sftp_files(sftp_folder=secrets.SFTP_FOLDER)
    dataframe = erap_loader.read_csv_into_dataframe('ERAP_PAYMENTS.csv', secrets.ERAP_DATA_TYPES)

    #: Update the AGOL data
    erap_logger.info('Updating data in AGOL')
    erap_updater = FeatureServiceInLineUpdater(dataframe, 'zip5')
    rows_updated = erap_updater.update_feature_service(
        secrets.ERAP_FEATURE_SERVICE_URL, list(secrets.ERAP_DATA_TYPES.keys())
    )

    #: Reclassify the break values on the webmap's color ramp
    erap_logger.info('Reclassifying the map')
    erap_reclassifier = ColorRampReclassifier(erap_webmap_item, gis)
    success = erap_reclassifier.update_color_ramp_values(secrets.ERAP_LAYER_NAME, 'Amount')

    reclassifier_result = 'Success'
    if not success:
        reclassifier_result = 'Failure'

    summary_message = MessageDetails()
    summary_message.subject = 'ERAP Update Summary'
    summary_message.message = f'{files_downloaded} files downloaded from SFTP\n{rows_updated} rows updated in Feature Service\n{reclassifier_result} of reclassifier.'

    erap_supervisor.notify(summary_message)


if __name__ == '__main__':
    #: the code that executes if you run the file or module directly
    process()
