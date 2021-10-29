#!/usr/bin/env python
# * coding: utf8 *
"""
test_projectname.py
A module that contains tests for the project module.
"""

from projectname import main


def test_hello_returns_hi():
    assert main.hello() == 'hi'
