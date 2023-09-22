#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
setup.py
A module that installs palletjack as a module
"""
from pathlib import Path

from setuptools import find_packages, setup

version = {}
with open('src/palletjack/version.py', encoding='utf-8') as fp:
    exec(fp.read(), version)

setup(
    name='ugrc-palletjack',
    version=version['__version__'],
    description='Updating AGOL feature services with data from external tables.',
    long_description=(Path(__file__).parent / 'README.md').read_text(),
    long_description_content_type='text/markdown',
    author='Jake Adams, UGRC',
    author_email='jdadams@utah.gov',
    url='https://github.com/agrc/palletjack',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    zip_safe=True,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Topic :: Utilities',
    ],
    project_urls={
        'Issue Tracker': 'https://github.com/agrc/palletjack/issues',
    },
    keywords=['gis'],
    install_requires=[
        'pysftp==0.2.9',
        'arcgis==2.1.*',
        'pyogrio==0.6.*',
        'pygsheets==2.0.*',
        'geopandas==0.14.*',
        'SQLAlchemy==1.4.*',
        'pg8000==1.29.*',
        'psycopg2-binary==2.9.*',
        'numpy<=1.23.5',  #: Pinned to < 1.24 to fix "module 'numpy' has no attribute 'str'" error
    ],
    extras_require={
        'nopro': [
            #: pinned versions to avoid backtracking if pro (and thus arcgis) aren't already installed
            'jupyterlab==3.6.5',
            'jupyter-server==1.24.0',
            'ipykernel==6.24.0',
            'ypy-websocket==0.8.2'
        ],
        'tests': [
            'pylint-quotes==0.2.*',
            'pylint==2.15.*',
            'pytest-cov==4.0.*',
            'pytest-instafail==0.4.*',
            'pytest-isort==3.1.*',
            'pytest-pylint==0.19.*',
            'pytest-watch==4.2.*',
            'pytest==7.*',
            'yapf==0.32.*',
            'pytest-mock==3.10.*',
        ]
    },
    setup_requires=[
        'pytest-runner',
    ],
    entry_points={'console_scripts': [
        'palletjack = palletjack.example:process',
    ]},
)
