# Overview

`palletjack` is a container package with multiple classes for updating data in AGOL based on information from external data. It handles the repetitive parts of the "extract" and "load" stages of the ETL process, allowing the user to focus on the custom transform steps unique to every project.

`palletjack` is not meant to be used by itself. Rather, its classes should be instantiated and used by other apps written for specific use cases. These other apps are referred to as "skids" or "client" apps.

Each class handles one step of the process and has public methods for accomplishing its task. There may be multiple, similar methods depending on exactly how you want to accomplish the task, or different methods for different types of data input or output - you probably won't use all the available methods in every application. These public methods usually call several private methods to keep functions small and testable.

pandas dataframes are the main unifying data structure between the different steps. "Loaders" handle the extract stage, pulling data in from external sources. Tabular data (csvs, Google Sheets) are loaded into dataframes, while non-tabular data are just downloaded to the specified locations.

Feature-service related "updaters" handle the ETL load stage for data headed to AGOL. After a skid transforms the dataframes it got from the loaders, the updaters use the dataframe as the input for modifying AGOL data.

Some loaders and updaters don't nicely fit in these patterns, but they still match the general idea of "downloading data" from an external source and then "updating/modifying" something on the AGOl side.

## Loaders

### SFTPLoader

The `SFTPLoader` class allows you to pull data from an SFTP share and load it into a pandas dataframe.

The initializer takes five arguments:

1. `host`: sftp hostname or IP address
1. `username`: sftp username
1. `password`: sftp password
1. `knownhosts_file`: path to a [knownhosts file](https://stackoverflow.com/questions/38939454/verify-host-key-with-pysftp) containing the public key of your sftp server
1. `download_dir`: the local directory to save the files in

#### Methods

- `download_sftp_folder_contents(self, sftp_folder='upload')`
  - Downloads all the files in `sftp_folder` to `self.download_dir`. Returns the number of files downloaded.
- `download_sftp_single_file(self, filename, sftp_folder='upload')`
  - Downloads `filename` from `sftp_folder` to `self.download_dir`. Returns `Path` object to the downloaded file.
- `read_csv_into_dataframe(self, filename, column_types=None)`
  - Reads `filename` from `self.download_dir` into a pandas dataframe, using optional `column_types` to define the column names and types. Returns the dataframe.
  - `column_types` is a dict of format `{'column_name': <type>}`, where `<type>` is `np.float64`, `str`, etc.

### GSheetLoader

`GSheetLoader` allows you to import some or all of the worksheets within a Google Sheets document into a pandas dataframe.

The initializer requires the path to a service account .json file that has access to the sheet in question. This account and file may need to be created by the GCP gurus.

#### Methods

- `load_specific_worksheet_into_dataframe(self, sheet_id, worksheet, by_title=False)`
  - Load a single worksheet from sheet_id into a dataframe, specified either with a 0-based index or the worksheet title.
- `load_all_worksheets_into_dataframes(self, sheet_id)`
  - Loads all the worksheets in sheet_id into a dictionary of dataframes, where the keys are the worksheet titles.
- `combine_worksheets_into_single_dataframe(self, worksheet_dfs)`
  - Attempts to merge all the worksheets in a dictionary (from `load_all_worksheets_into_dataframes()`) into a single dataframe, adding a 'worksheet' column to identify the worksheet each row came from/belongs to. Will raise an error if all the dataframes don't have the same columns.

### GoogleDriveDownloader

`GoogleDriveDownloader` provides methods to download any non-html file (ie, Content-Type != text/html) Google Drive file from it's sharing link. The files may be publicly shared or shared with a service account.

The initializer sets the output directory for saving the files. The `out_dir` attribute can be modified at any time to change this destination.

This class has two similar sets of methods. The `*_using_api` methods authenticate to the Google API using a service account file and download using the API. If at all possible, use these methods to have the least likelihood of errors. They support publicly shared files along with files just shared to the service account.

The other methods just use an anonymous HTTP request, which requires the files to be shared publicly. Google will block you from downloading after a certain amount of time/requests. A pause has been added to overcome this, but I've yet to find a good value for it.

#### Methods

- `download_file_from_google_drive_using_api(self, gsheets_client, sharing_link, join_id)`
  - Download a file to the out_dir set on the instantiated object using the Google API. `gsheets_client` is the authenticated Client object from `pygsheets.authorize()`. `sharing_link` should be in the form `https://drive.google.com/file/d/big_long_id/etc`. `join_id` is used for logging purposes to identify which attachment is being worked on. Will log an error if the URL doesn't match this pattern or it can't extract the unique id from the sharing URL. Will also log an error if the header's Content-Type is text/html, which usually indicates the HTTP response was an error message instead of the file.
- `download_attachments_from_dataframe_using_api(self, service_file, dataframe, sharing_link_column, join_id_column, output_path_column)`
  - Calls `download_file_from_google_drive_using_api` for every row in `dataframe`, saving the full path of the resulting file in `output_path_column`. Uses `service_file` to authenticate to the Google API using `pygsheets.authorize()`
- `download_file_from_google_drive(self, sharing_link, join_id, pause=0.)`
  - Download a file to the out_dir set on the instantiated object using an anonymous HTTP request. `sharing_link` should be in the form `https://drive.google.com/file/d/big_long_id/etc`. `join_id` is used for logging purposes to identify which attachment is being worked on. `pause` specifies a sleep in seconds before downloading to try to get around Google's blocking. Will log an error if the URL doesn't match this pattern or it can't extract the unique id from the sharing URL. Will also log an error if the header's Content-Type is text/html, which usually indicates the HTTP response was an error message instead of the file.
- `download_attachments_from_dataframe(self, dataframe, sharing_link_column, join_id_column, output_path_column)`
  - Calls `download_file_from_google_drive` for every row in `dataframe`, saving the full path of the resulting file in `output_path_column`

## Updaters

### FeatureServiceInlineUpdater

Updates the attributes of existing features in an AGOL Hosted Feature Service using either the `arcpy` or `arcgis` libraries. The `arcgis` library is required for interacting with AGOL.

The initializer has three required arguments and one optional argument:

1. `gis`: An `arcgis.gis.GIS` object representing your AGOL organization.
1. `dataframe`: A pandas dataframe containing the new data where each row is a separate record.
1. `index_column`: The index column that is present in both the existing and new data for joining the datasets.
1. `field_mapping`(optional): A dictionary of existing field names to new field names for matching the new data to the existing data in AGOL.

#### Methods

- `update_existing_features_in_feature_service_with_arcpy(self, feature_service_url, fields)`
  - Update `fields` in `feature_service_url` with data from `self.new_dataframe()` using `arcpy.da.UpdateCursor`. Requires arcpy to be installed. Returns the number of rows updated.
- `update_existing_features_in_hosted_feature_layer(self, feature_layer_itemid, fields)`
  - update `fields` in `self.gis`'s `feature_layer_itemid` item with data from `self.new_dataframe()` using `arcgis.feature.FeatureLayer.edit_features()`. Returns the number of rows successfully updated. If any updates fail, it tries to roll back all successful updates.
- `upsert_new_data_in_hosted_feature_layer(self, feature_service_item_id, layer_index=0)`
  - UPdate existing data and inSERT new data into feature service referenced by `feature_service_item_id` (at `layer_index` within the feature service) with data from `self.new_dataframe()` using `arcgis.features.FeatureLayer.append()`. The new data must not include any fields not present in the live data. The index column in the hosted feature service defined by `self.index_column` must be marked as 'Unique' in ArcGIS Online.
    - You can't set the field as unique if it was specified as the Display Field when it was uploaded from Pro (layer properties -> Display -> Display Field). You can change this value and overwrite the feature service to change the display field. There may be a way to do it without re-writing the service, but I've not found it yet.
    - I've had trouble specifying string fields as unique. You may have better luck using a numeric field as `self.index_column`.

### FeatureServiceOverwriter

Completely overwrite a feature service rather than updating it feature-by-feature.

The initializer requires the `arcgis.gis.GIS` object representing your organization.

#### Methods

- `truncate_and_load_feature_service(self, feature_service_item_id, new_dataframe, failsafe_dir, layer_index=0)`
  - Attempts to delete existing data from a feature layer and add new data from a spatially-enabled dataframe. First attempts to truncate existing data and loads it in memory as a dataframe. Then renames new data column names to conform to AGOL scheme (spaces, special chars changed to '_'). Finally attempts to append new data to now-empty feature layer using `arcgis.features.FeatureLayer.append()`. If the new data append fails, it attempts to re-upload the previous data from the in-memory dataframe. If this fails, it attempts to failsafe by writing the old data to disk as a json file.

### FeatureServiceAttachmentsUpdater

Updates the attachments on a hosted feature service based on a dataframe containing two columns: a join key present in the live data (the dataframe column name must match the feature service field name) and the file to attach to the feature. While AGOL supports multiple attachments, this only accepts a single file as input.

If a matching feature in AGOl doesn't have an attachment, the file referred to by the dataframe will be uploaded. If it does have an attachment, it checks the existing filename with the referenced file. If they are different, the file from the dataframe will be updated. If they are the same, nothing happens.

The initializer requires the `arcgis.gis.GIS` object representing your organization.

#### Methods

- `update_attachments(self, feature_layer_itemid, attachment_join_field, attachment_path_field, attachments_df, layer_number=0)`
  - Updates the attachments on layer `layer_number` of `feature_layer_itemid`. Requires a dataframe containing one row for each feature to have it's attachments updated. This dataframe must have an `attachment_join_field` containing the unique key linking the live data to the dataframe and an `attachment_path_field` containing the path to the new attachment.

### ColorRampReclassifier

Manually edits a webmap's JSON definition to change a layer's color ramp values based on a simple unclassed scheme similar to AGOL's unclassed ramp. The minimum value is the dataset minimum, the max is the mean value plus one standard deviation.

The initializer only takes two arguments:

1. `webmap_item`: The `arcgis.gis.Item` object of the webmap to update.
1. `gis`: The `arcgis.gis.GIS` object of the AGOL organization.

#### Methods

- `update_color_ramp_values(self, layer_name, column_name, stops=5)`
  - Updates the color ramp stop values for `column_name` of `layer_name` in `self.webmap_item` using `stops` number of stops. Returns a boolean indicating success (passed through from `self.webmap_item.update()` call).

## Logging

`palletjack` takes full advantage of python's built-in `logging` library to perform its feedback and reporting. Each class creates it's own child logger under the main `palletjack` logger, becoming `palletjack.SFTPLoader`, etc.

### Accessing `palletjack` logs

Client programs can get a reference to the `palletjack` logger and add their handlers, formatters, etc to it alongside the client's logger:

```python
myapp_logger = logging.getLogger('my_app')
myapp_logger.setLevel(secrets.LOG_LEVEL)
palletjack_logger = logging.getLogger('palletjack')
palletjack_logger.setLevel(secrets.LOG_LEVEL)
#: set up handlers and formatters
#: ...
myapp_logger.addHandler(log_handler)
palletjack_logger.addHandler(log_handler)
```

### Log levels

- `logging.DEBUG` includes verbose debug info that should allow you to manually (possibly programmatically) undo or redo any operation.
- `logging.INFO` includes standard runtime progress reports and result information.
- `logging.WARNING` includes negative results from checks or other situations that the user should be aware of.

## Exception Handling

Some exceptions are captured and [chained](https://docs.python.org/3/tutorial/errors.html#exception-chaining) to include additional, context-specific messages for the user. The client is responsible for handling all re-raised exceptions and any other errors.
