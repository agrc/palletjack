#!/usr/bin/env python
# * coding: utf8 *
"""
a description of what this module does.
this file is for testing linting...
"""

from supervisor.message_handlers import SendGridHandler
from supervisor.models import MessageDetails, Supervisor

from . import secrets, updates

era_supervisor = Supervisor()

era_supervisor.add_message_handler(SendGridHandler(secrets.SENDGRID_SETTINGS, 'ERA'))

if __name__ == '__main__':
    #: the code that executes if you run the file or module directly
    pass
