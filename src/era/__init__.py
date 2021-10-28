import logging
import sys
from logging.handlers import RotatingFileHandler

from supervisor.message_handlers import SendGridHandler
from supervisor.models import Supervisor

from . import secrets

LOG_LEVEL = logging.INFO

package_logger = logging.getLogger(__name__)
package_logger.setLevel(LOG_LEVEL)
cli_handler = logging.StreamHandler(sys.stdout)
cli_handler.setLevel(LOG_LEVEL)
formatter = logging.Formatter(
    fmt='%(levelname)-7s %(asctime)s %(module)15s:%(lineno)5s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
)
cli_handler.setFormatter(formatter)
package_logger.addHandler(cli_handler)

log_handler = RotatingFileHandler(secrets.ERAP_LOG_PATH, backupCount=secrets.ROTATE_COUNT)
#: Rotate the log on each run (also gets rotated on each import, so each save in VS Code rotates too...)
#: Putting stuff in here also causes it all to be run on test discovery, too. Maybe not the best place for all this...
log_handler.doRollover()
log_handler.setLevel(LOG_LEVEL)
log_handler.setFormatter(formatter)
package_logger.addHandler(log_handler)

package_logger.debug('Creating Supervisor object')
erap_supervisor = Supervisor(logger=package_logger, log_path=secrets.ERAP_LOG_PATH)
erap_supervisor.add_message_handler(SendGridHandler(sendgrid_settings=secrets.SENDGRID_SETTINGS, project_name='era'))
