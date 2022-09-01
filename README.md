# agrc/palletjack

![Build Status](https://github.com/agrc/palletjack/workflows/Build%20and%20Test/badge.svg)
[![codecov](https://codecov.io/gh/agrc/palletjack/branch/main/graph/badge.svg)](https://codecov.io/gh/agrc/palletjack)
<!-- 1. Navigate to [codecov.io](https://codecov.io/gh/agrc/python) and create a `CODECOV_TOKEN` [project secret](https://github.com/agrc/python/settings/secrets) -->

A library of classes for automatically updating AGOL feature services with data from external sources. Client apps (often called 'skids') can reuse these classes for common use cases. These classes handle different parts of the Extract and Load steps in the ETL process. Tabular data is loaded into dataframes, which are then processed by the skid and passed to the updater classes that update the feature service.

See `docs/api.md` for documentation on the available classes and methods, logging, and errors.

See `docs/examples.py` for (bare-bones) example code implementing the various classes/methods. You can also search our GitHub organization for "skid" repositories that use palletjack.

Pallet jack: [forklift's](https://www.github.com/agrc/forklift) little brother.

## Installation

1. Activate your application's environment
1. `pip install ugrc-palletjack`

## Dependencies

`palletjack` relies on `setup.py` to install `pandas`, `numpy`, `pysftp`, and `arcgis`. `FeatureServiceInlineUpdater.update_existing_features_in_feature_service_with_arcpy()` also relies on having arcpy installed through either ArcGIS Pro or ArcGIS Enterprise.

## Usage

1. `import palletjack`
1. Instantiate objects as needed:

   ```python
   loader = palletjack.SFTPLoader(secrets, download_dir)
   files_downloaded = loader.download_sftp_files(sftp_folder=secrets.SFTP_FOLDER)
   dataframe = loader.read_csv_into_dataframe('data.csv', secrets.DATA_TYPES)

   updater = FeatureServiceInlineUpdater(gis, dataframe, secrets.KEY_COLUMN)
   rows_updated = updater.update_existing_features_in_hosted_feature_layer(
      secrets.FEATURE_LAYER_ITEMID, list(secrets.DATA_TYPES.keys())
    )
   ```

## Development

1. Create a conda environment with arcpy, arcgis
   - `conda create -n palletjack`
   - `activate palletjack`
   - `conda install arcgis arcpy -c esri`
1. Clone the repo
1. Install in dev mode
   - `pip install -e .[tests]`

### Updating pypi

1. Delete everything in dist/
1. Make sure you've updated the version number in setup.py
1. Recreate the wheels:
   - python setup.py sdist bdist_wheel
1. Re-upload the new files
   - twine upload dist/*
