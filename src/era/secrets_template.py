"""
secrets_template.py: Example secrets file. Copy to 'secrets.py' and populate with actual values.
DO NOT ADD new secrets.py to version control.
"""

import socket
from pathlib import Path

import numpy as np

AGOL_ORG = 'https://utah.maps.arcgis.com'
AGOL_USER = ''
AGOL_PASSWORD = ''
SENDGRID_SETTINGS = {  #: Settings for SendGridHandler
    'api_key': '',
    'from_address': 'noreply@utah.gov',
    'to_addresses': '',
    'prefix': f'ERA on {socket.gethostname()}: ',
}
ROTATE_COUNT = 90

SFTP_HOST = ''
SFTP_USERNAME = ''
SFTP_PASSWORD = ''
SFTP_FOLDER = ''
KNOWNHOSTS = f'{Path(__file__).parent.parent.parent}\\known_hosts'

ERAP_FEATURE_SERVICE_URL = ''
ERAP_WEBMAP_ITEMID = ''
ERAP_LAYER_NAME = ''
ERAP_BASE_DIR = Path()
ERAP_LOG_PATH = ERAP_BASE_DIR / 'log.txt'
ERAP_DATA_TYPES = {
    'zip5': str,
    'Count_': str,
    'Amount': np.float64,
    'Updated': str,
}
