"""Errors specific to palletjack"""


class IntFieldAsFloatError(Exception):
    """A field expected to be integer is float instead"""


class TimezoneAwareDatetimeError(Exception):
    """A datetime field uses a timezone-aware dtype"""
