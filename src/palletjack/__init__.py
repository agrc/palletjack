"""A library for extracting tabular/spatial data from many different sources, transforming it to meet ArcGIS Online hosted feature service requirements, and loading it into existing feature services.

.. include:: ../../docs/README.md
"""

import locale

from . import extract, load, transform, utils  # noqa: F401
from .errors import IntFieldAsFloatError, TimezoneAwareDatetimeError  # noqa: F401

#: If the locale number category is not set explicitly, set it to en_US and UTF-8 encoding to facilitate parsing
#: strings with comma separators to numeric types
if not locale.getlocale(locale.LC_NUMERIC)[0]:
    locale.setlocale(locale.LC_NUMERIC, ("en_US", "UTF-8"))
