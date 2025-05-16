# palletjack

[![Release Events](https://github.com/agrc/palletjack/actions/workflows/release.yml/badge.svg)](https://github.com/agrc/palletjack/actions/workflows/release.yml)
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
   updater = load.ServiceUpdater(gis, 'feature_service_item_id')
   updates = updater.truncate_and_load(cleaned_df)

   #: It even works with stand-alone tables!
   table_updater = load.TableUpdater(gis, 'table_service_item_id', service_type='table')
   table_updates = table_updater.truncate_and_load(cleaned_df)
   ```

## Development

1. Create a conda environment with Python 3.11
   - `conda create -n palletjack python=3.11`
   - `activate palletjack`
1. Clone the repo
1. Install in dev mode with development dependencies
   - `pip install -e .[tests]`

### Troubleshooting Weird Append Errors

If a `FeatureLayer.append()` call (within a load.ServiceUpdater method) fails with an "Unknown Error: 500" error or something like that, you can query the results to get more info. The `urllib3` debug-level logs will include the HTTP GET or POST call, something like the following:
`https://services1.arcgis.com:443 POST /<unique string>/arcgis/rest/services/<feature layer name>/FeatureServer/<layer id>/append/jobs/<job guid>?f=json token=<crazy long token string>`. The defualt `basicConfig` logger includes the `urllib3` logs (`logging.basicConfig(level=logging.DEBUG)`) and is great for development debugging, or you can add a specific logger for `urllib3` in your code and set it's level to debug.

You can use this and a token from an AGOL tab to build a new job status url. To get the token, log into AGOL in a browser and open a private hosted feature layer item. Click the layer, and then open the developer console. With the Network tab of the console open, click on the "View" link for the service URL. You should see a document in the list whose name includes "?token=<really long token string>". Copy the name and then copy out the token string.

Now that you've got the token string, you can build the status query:
`https://services1.arcgis.com/<unique string>/arcgis/rest/services/<feature layer name>/FeatureServer/<layer id>/append/jobs/<job guid>?f=json&<token from agol>`

Calling this URL in a browser should return a message that will hopefully give you more info as to why it failed.

### Updating Docs

palletjack uses `pdoc3` to generate HTML docs in `docs/palletjack` from the docstrings within the code itself. These are then served up via github pages.

The github pages are served from the `gh-pages` branch. After you make edits to the code and update the docstrings, rebase this branch onto the updated `main` branch. To prevent github pages from trying to generate a site from the contents of `docs/palletjack` with jekyll, add a `.nojekyll` file to `docs/palletjack`.

To generate the docs, run `pdoc --html -o docs\ c:\palletjack\repo\src\palletjack --force`. The code's docstrings should be Google-style docstrings with proper indentation to ensure the argument lists, etc are parsed and displayed correctly.

`docs/README.md` is included at the top package level by adding the line `.. include:: ../../docs/README.md` in `__init__.py`'s docstring. This tells pdoc to insert that markdown into the HTML generated for that docstring, and the include directive can be used for more in-depth documentation anywhere else as well. Note that `pdoc` tries to create links for anything surrounded by backticks, which are also used for code blocks. You may need to manually edit the HTML to remove the links if they change the content of your code blocks (such as in the example import statement).

Once the contents of `docs/palletjack` look correct, force push the `gh-pages` branch to github. This will trigger the action to republish the site. The docs are then accessible at [agrc.github.io/palletjack/palletjack/index.html].
