#!/usr/bin/env python
# * coding: utf8 *
"""
Updates the DWS ERAP layer based on their weekly
"""

from pathlib import Path

import arcgis
import numpy as np
from supervisor.message_handlers import SendGridHandler
from supervisor.models import MessageDetails, Supervisor

from . import secrets
from .data_coupler import ColorRampReclassifier, FeatureServiceInLineUpdater, SFTPLoader


def process():
    erap_supervisor = Supervisor()

    erap_supervisor.add_message_handler(SendGridHandler(secrets.SENDGRID_SETTINGS, 'ERA'))

    erap_download_dir = Path()
    erap_data_types = {
        'ZipCode': str,
        'Count': str,
        'Amount': np.float64,
        'LastUpdated': str,
    }
    erap_feature_service_url = ''
    gis = arcgis.gis.GIS(secrets.AGOL_ORG, secrets.AGOL_USER, secrets.AGOL_PASSWORD)
    erap_webmap_item = gis.content.get('')
    erap_layer_name = ''

    #: Load the latest data from FTP
    erap_loader = SFTPLoader(secrets, erap_download_dir)
    erap_loader.download_sftp_files()
    dataframe = erap_loader.read_csv_into_dataframe('ERAP_PAYMENTS.csv', erap_data_types)

    #: Update the AGOL data
    erap_updater = FeatureServiceInLineUpdater(dataframe, 'ZipCode')
    erap_updater.update_feature_service(erap_feature_service_url, ['Count', 'Amount', 'LastUpdated'])

    #: Reclassify the break values on the webmap's color ramp
    erap_reclassifier = ColorRampReclassifier(erap_webmap_item, gis)
    success = erap_reclassifier.update_color_ramp_values(erap_layer_name, 'Amount')


if __name__ == '__main__':
    #: the code that executes if you run the file or module directly
    process()
