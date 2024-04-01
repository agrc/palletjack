"""A library for extracting tabular/spatial data from many different sources, transforming it to meet ArcGIS Online hosted feature service requirements, and loading it into existing feature services.

.. include:: ../../docs/README.md
"""

import locale

from . import extract, load, transform, utils  # noqa: F401
from .errors import IntFieldAsFloatError, TimezoneAwareDatetimeError  # noqa: F401

#: If the locale is not set explicitly, set it to the system default for text to number conversions
if not locale.getlocale(locale.LC_NUMERIC)[0]:
    locale.setlocale(locale.LC_NUMERIC, locale.getlocale())
