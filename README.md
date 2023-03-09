# agrc/palletjack

![Build Status](https://github.com/agrc/palletjack/workflows/Build%20and%20Test/badge.svg)
[![codecov](https://codecov.io/gh/agrc/palletjack/branch/main/graph/badge.svg)](https://codecov.io/gh/agrc/palletjack)

A library of classes and methods for automatically updating AGOL feature services with data from several different types of external sources. Client apps (sometimes called 'skids') can reuse these classes for common use cases. The code modules are oriented around each step in the extract, transform, and load process.

`palletjack` works with pandas DataFrames (either regular for tabular data or Esri's spatially-enabled dataframes for spatial data). The extract and transform methods return dataframes and the load methods consume dataframes as their source data.

The [documentation](https://agrc.github.io/palletjack/palletjack) includes a user guide along with an API description of the available classes and methods.

Pallet jack: [forklift's](https://www.github.com/agrc/forklift) little brother.

## Dependencies

`palletjack` relies on the dependencies listed in `setup.py`. These are all available on PyPI and can be installed in most environments, including Google Cloud Functions.

The `arcgis` library does all the heavy lifting for spatial data. If the `arcpy` library is not available (such as in a cloud function), it relies on `shapely` for its geometry engine.

## Installation

1. Activate your application's environment
1. `pip install ugrc-palletjack`

## Quick start

1. Import the desired modules
1. Use a class in `extract` to load a dataframe from an external source
1. Transform your dataframe as desired with helper methods from `transform`
1. Use the dataframe to update a hosted feature service using the methods in `load`

   ```python
   from palletjack import extract, transform, load

   #: Load the data from a Google Sheet
   gsheet_extractor = extract.GSheetLoader(path_to_service_account_json)
   sheet_df = gsheet_extractor.load_specific_worksheet_into_dataframe(sheet_id, 'title of desired sheet', by_title=True)

   #: Convert the data to points using lat/long fields, clean for uploading
   spatial_df = pd.DataFrame.spatial.from_xy(input_df, x_column='longitude', y_column='latitude')
   renamed_df = transform.DataCleaning.rename_dataframe_columns_for_agol(spatial_df)
   cleaned_df = transform.DataCleaning.switch_to_nullable_int(renamed_df, ['an_int_field_with_null_values'])

   #: Truncate the existing feature service data and load the new data
   gis = arcgis.gis.GIS('my_agol_org_url', 'username', 'super-duper-secure-password')
   updates = palletjack.load.FeatureServiceUpdater.truncate_and_load_features(
      gis, 'feature_service_item_id', cleaned_df, r'c:\directory\to\save\truncated\data\in\case\of\error'
   )
   ```

## Development

1. Create a conda environment with Python 3.9
   - `conda create -n palletjack python=3.9`
   - `activate palletjack`
1. Clone the repo
1. Install in dev mode with development dependencies
   - `pip install -e .[tests]`

### Troubleshooting Weird Append Errors

If a `FeatureLayer.append()` call (within a load.FeatureServiceUpdater method) fails with an "Unknown Error: 500" error or something like that, you can query the results to get more info. The debug log will include the HTTP GET call, something like the following:
`https://services1.arcgis.com:443 POST /<unique string>/arcgis/rest/services/<feature layer name>/FeatureServer/<layer id>/append/jobs/<job guid>?f=json token=<crazy long token string>`

You can use this and a token from an AGOL tab to build a new job status url. To get the token, log into AGOL in a browser and open a private hosted feature layer item. Click the layer, and then open the developer console. With the Network tab of the console open, click on the "View" link for the service URL. You should see a document in the list whose name includes "?token=<really long token string>". Copy the name and then copy out the token string.

Now that you've got the token string, you can build the status query:
`https://services1.arcgis.com/<unique string>/arcgis/rest/services/<feature layer name>/FeatureServer/<layer id>/append/jobs/<job guid>?f=json&<token from agol>`

Calling this URL in a browser should return a message that will hopefully give you more info as to why it failed.
