# agrc/palletjack

![Build Status](https://github.com/agrc/palletjack/workflows/Build%20and%20Test/badge.svg)
[![codecov](https://codecov.io/gh/agrc/palletjack/branch/main/graph/badge.svg)](https://codecov.io/gh/agrc/palletjack)

A library of classes for automatically updating AGOL feature services with data from external sources. Client apps (often called 'skids') can reuse these classes for common use cases. These classes handle different parts of the Extract and Load steps in the ETL process.

`palletjack` works with pandas DataFrames (either regular for tabular data or Esri's spatially-enabled dataframes for spatial data). Most methods either return a dataframe or use a dataframe for their source data.

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
1. Instantiate one or more of the classes as needed.
1. Call the methods on your instantiated objects to perform the specific action desired.

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

### Troubleshooting Weird Append Errors

If a `FeatureLayer.append()` call (within a load.FeatureServiceUpdater method) fails with an "Unknown Error: 500" error or something like that, you can query the results to get more info. The debug log will include the HTTP GET call, something like the following:
`https://services1.arcgis.com:443 POST /<unique string>/arcgis/rest/services/<feature layer name>/FeatureServer/<layer id>/append/jobs/<job guid>?f=json token=<crazy long token string>`

You can use this and a token from an AGOL tab to build a new job status url. To get the token, log into AGOL in a browser and open a private hosted feature layer item. Click the layer, and then open the developer console. With the Network tab of the console open, click on the "View" link for the service URL. You should see a document in the list whose name includes "?token=<really long token string>". Copy the name and then copy out the token string.

Now that you've got the token string, you can build the status query:
`https://services1.arcgis.com/<unique string>/arcgis/rest/services/<feature layer name>/FeatureServer/<layer id>/append/jobs/<job guid>?f=json&<token from agol>`

Calling this URL in a browser should return a message that will hopefully give you more info as to why it failed.
