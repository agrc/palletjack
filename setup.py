#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
setup.py
A module that installs palletjack as a module
"""

from pathlib import Path

from setuptools import find_packages, setup

version = {}
with open("src/palletjack/version.py", encoding="utf-8") as fp:
    exec(fp.read(), version)

setup(
    name="ugrc-palletjack",
    version=version["__version__"],
    description="Updating AGOL feature services with data from external tables.",
    long_description=(Path(__file__).parent / "README.md").read_text(),
    long_description_content_type="text/markdown",
    author="Jake Adams, UGRC",
    author_email="jdadams@utah.gov",
    url="https://github.com/agrc/palletjack",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    zip_safe=True,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Topic :: Utilities",
    ],
    project_urls={
        "Issue Tracker": "https://github.com/agrc/palletjack/issues",
    },
    keywords=["gis"],
    install_requires=[
        "arcgis>=2.3,<2.5",
        "geopandas>=0.14,<1.2",
        "geodatasets>=2023.12,<2024.9",
        "pg8000>=1.29,<1.32",
        "psycopg2-binary==2.9.*",
        "pygsheets==2.0.*",
        "pysftp==0.2.9",
        "SQLAlchemy>=1.4,<2.1",
        # Temporary pin to override pysftp's dependency resolution.
        # TODO: Migrate away from pysftp to use paramiko directly. See https://github.com/agrc/palletjack/issues/123
        "paramiko<4.0.0",
    ],
    extras_require={
        "tests": [
            "pdoc3>=0.10,<0.12",
            "pytest-cov>=3,<7",
            "pytest-instafail~=0.4",
            "pytest-mock>=3.10,<3.15",
            "pytest-watch~=4.2",
            "pytest>=6,<9",
            "requests-mock==1.*",
            "ruff==0.*",
        ]
    },
    setup_requires=[
        "pytest-runner",
    ],
    entry_points={
        "console_scripts": [
            "palletjack = palletjack.example:process",
        ]
    },
)
