import logging
import sys

from supervisor.message_handlers import SendGridHandler
from supervisor.models import Supervisor

from . import secrets

package_logger = logging.getLogger(__name__)
package_logger.setLevel(logging.DEBUG)
cli_handler = logging.StreamHandler(sys.stdout)
cli_handler.setLevel(logging.DEBUG)
cli_formatter = logging.Formatter(
    fmt='%(levelname)-7s %(asctime)s %(module)10s:%(lineno)5s %(message)s', datefmt='%m-%d %H:%M:%S'
)
cli_handler.setFormatter(cli_formatter)
package_logger.addHandler(cli_handler)

package_logger.debug('Creating Supervisor object')
erap_supervisor = Supervisor(logger=package_logger)
erap_supervisor.add_message_handler(SendGridHandler(sendgrid_settings=secrets.SENDGRID_SETTINGS, project_name='era'))
