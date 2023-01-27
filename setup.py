#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
setup.py
A module that installs palletjack as a module
"""
from glob import glob
from os.path import basename, splitext

from setuptools import find_packages, setup

setup(
    name='ugrc-palletjack',
    version='3.0.0.beta1',
    license='MIT',
    description='Updating AGOL feature services with data from SFTP shares.',
    author='Jake Adams, UGRC',
    author_email='jdadams@utah.gov',
    url='https://github.com/agrc/palletjack',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=True,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Utilities',
    ],
    project_urls={
        'Issue Tracker': 'https://github.com/agrc/palletjack/issues',
    },
    keywords=['gis'],
    install_requires=[
        'pysftp==0.2.9',
        'arcgis==2.1.*',
        'pygsheets==2.0.*',
        'geopandas==0.12.*',
        'SQLAlchemy==1.4.*',
        'pg8000==1.29.*',
        'psycopg2-binary==2.9.*',
        'numpy==1.24.*',  #: May need to pin to 1.23.* to fix "module 'numpy' has no attribute 'str'" error
    ],
    extras_require={
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
