"""
secrets_template.py: Example secrets file. Copy to 'secrets.py' and populate with actual values.
DO NOT ADD new secrets.py to version control.
"""

from pathlib import Path

SFTP_HOST = ''
SFTP_USERNAME = ''
SFTP_PASSWORD = ''
KNOWNHOSTS = f'{Path(__file__).parent.parent.parent}\\known_hosts'
