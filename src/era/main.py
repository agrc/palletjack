#!/usr/bin/env python
# * coding: utf8 *
"""
Updates the DWS ERAP layer based on their weekly
"""

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

import arcgis
from supervisor.message_handlers import SendGridHandler
from supervisor.models import MessageDetails, Supervisor

from . import secrets
from .data_coupler import ColorRampReclassifier, FeatureServiceInLineUpdater, SFTPLoader


def _make_download_dir(exist_ok=False):
    today = datetime.today()
    download_dir = secrets.ERAP_BASE_DIR / today.strftime('%Y%m%d_%H%M%S')
    try:
        download_dir.mkdir(exist_ok=exist_ok)
    except FileNotFoundError as error:
        raise FileNotFoundError(f'Base directory {secrets.ERAP_BASE_DIR} does not exist.') from error
    else:
        return download_dir


def _initialize(log_level):

    erap_logger = logging.getLogger('era')
    erap_logger.setLevel(log_level)
    cli_handler = logging.StreamHandler(sys.stdout)
    cli_handler.setLevel(log_level)
    formatter = logging.Formatter(
        fmt='%(levelname)-7s %(asctime)s %(module)15s:%(lineno)5s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )
    cli_handler.setFormatter(formatter)
    erap_logger.addHandler(cli_handler)

    log_handler = RotatingFileHandler(secrets.ERAP_LOG_PATH, backupCount=secrets.ROTATE_COUNT)
    log_handler.doRollover()  #: Rotate the log on each run
    log_handler.setLevel(log_level)
    log_handler.setFormatter(formatter)
    erap_logger.addHandler(log_handler)

    erap_logger.debug('Creating Supervisor object')
    erap_supervisor = Supervisor(logger=erap_logger, log_path=secrets.ERAP_LOG_PATH)
    erap_supervisor.add_message_handler(
        SendGridHandler(sendgrid_settings=secrets.SENDGRID_SETTINGS, project_name='era')
    )

    return erap_supervisor


def process():

    start = datetime.now()

    erap_supervisor = _initialize(logging.INFO)
    # : Putting this down here so logging/supervisor catches any license issues
    import arcpy  # pylint: disable=import-outside-toplevel

    module_logger = logging.getLogger(__name__)

    module_logger.debug('Logging into `%s` as `%s`', secrets.AGOL_ORG, secrets.AGOL_USER)
    gis = arcgis.gis.GIS(secrets.AGOL_ORG, secrets.AGOL_USER, secrets.AGOL_PASSWORD)
    arcpy.SignInToPortal(secrets.AGOL_ORG, secrets.AGOL_USER, secrets.AGOL_PASSWORD)
    erap_webmap_item = gis.content.get(secrets.ERAP_WEBMAP_ITEMID)
    erap_download_dir = _make_download_dir()

    #: Load the latest data from FTP
    module_logger.info('Getting data from FTP')
    erap_loader = SFTPLoader(secrets, erap_download_dir)
    files_downloaded = erap_loader.download_sftp_files(sftp_folder=secrets.SFTP_FOLDER)
    dataframe = erap_loader.read_csv_into_dataframe('ERAP_PAYMENTS.csv', secrets.ERAP_DATA_TYPES)

    #: Update the AGOL data
    module_logger.info('Updating data in AGOL')
    erap_updater = FeatureServiceInLineUpdater(dataframe, 'zip5')
    rows_updated = erap_updater.update_feature_service(
        secrets.ERAP_FEATURE_SERVICE_URL, list(secrets.ERAP_DATA_TYPES.keys())
    )

    #: Reclassify the break values on the webmap's color ramp
    module_logger.info('Reclassifying the map')
    erap_reclassifier = ColorRampReclassifier(erap_webmap_item, gis)
    success = erap_reclassifier.update_color_ramp_values(secrets.ERAP_LAYER_NAME, 'Amount')

    reclassifier_result = 'Success'
    if not success:
        reclassifier_result = 'Failure'

    end = datetime.now()

    summary_message = MessageDetails()
    summary_message.subject = 'ERAP Update Summary'
    summary_rows = [
        f'ERAP update {start.strftime("%Y-%m-%d")}',
        '=' * 20,
        '',
        f'Start time: {start.strftime("%H:%M:%S")}',
        f'End time: {end.strftime("%H:%M:%S")}',
        f'Duration: {str(end-start)}',
        f'{files_downloaded} files downloaded from SFTP',
        f'{rows_updated} rows updated in Feature Service',
        f'Reclassifier webmap update operation: {reclassifier_result}',
    ]
    summary_message.message = '\n'.join(summary_rows)
    summary_message.attachments = secrets.ERAP_LOG_PATH

    erap_supervisor.notify(summary_message)


if __name__ == '__main__':
    #: the code that executes if you run the file or module directly
    process()
