# Overview

`palletjack` is a container package with multiple classes for updating data in AGOL based on information from an SFTP share. Each class handles one step of the process and has public methods for accomplishing its task. There may be multiple, similar methods depending on exactly how you want to accomplish the task. These public methods usually call several private methods to keep functions small and testable.

`palletjack` is not meant to be used by itself. Rather, its classes should be instantiated and used by other apps written for specific use cases. These other apps are referred to as the "client" apps.

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

## Classes

### SFTPLoader

The `SFTPLoader` class allows you to pull data from an SFTP share and load it into a pandas dataframe.

The initializer takes five arguments:

1. `host`: sftp hostname or IP address
1. `username`: sftp username
1. `password`: sftp password
1. `knownhosts_file`: path to a [knownhosts file](https://stackoverflow.com/questions/38939454/verify-host-key-with-pysftp) containing the public key of your sftp server
1. `download_dir`: the local directory to save the files in

Methods

- `download_sftp_folder_contents(self, sftp_folder='upload')`
  - Downloads all the files in `sftp_folder` to `self.download_dir`. Returns the number of files downloaded.
- `download_sftp_single_file(self, filename, sftp_folder='upload')`
  - Downloads `filename` from `sftp_folder` to `self.download_dir`. Returns `Path` object to the downloaded file.
- `read_csv_into_dataframe(self, filename, column_types=None)`
  - Reads `filename` from `self.download_dir` into a pandas dataframe, using optional `column_types` to define the column names and types. Returns the dataframe.
  - `column_types` is a dict of format `{'column_name': <type>}`, where `<type>` is `np.float64`, `str`, etc.

### FeatureServiceInlineUpdater

Updates the attributes of existing features in an AGOL Hosted Feature Service using either the `arcpy` or `arcgis` libraries. The `arcgis` library is required for interacting with AGOL.

The initializer takes three arguments:

1. `gis`: An `arcgis.gis.GIS` object representing your AGOL organization.
1. `dataframe`: A pandas dataframe containing the new data where each row is a separate record.
1. `index_column`: The index column that is present in both the existing and new data for joining the datasets.

Methods

- `update_existing_features_in_feature_service_with_arcpy(self, feature_service_url, fields)`
  - Update `fields` in `feature_service_url` with data from `self.new_dataframe()` using `arcpy.da.UpdateCursor`. Requires arcpy to be installed. Returns the number of rows updated.
- `update_existing_features_in_hosted_feature_layer(self, feature_layer_itemid, fields)`
  - update `fields` in `self.gis`'s `feature_layer_itemid` item with data from `self.new_dataframe()` using `arcgis.feature.FeatureLayer.edit_features()`. Returns the number of rows successfully updated. If any updates fail, it tries to roll back all successful updates.

### FeatureServiceOverwriter

Not implemented. Will completely overwrite a feature service rather than updating it feature-by-feature.

### ColorRampReclassifier

Manually edits a webmap's JSON definition to change a layer's color ramp values based on a simple unclassed scheme similar to AGOL's unclassed ramp. The minimum value is the dataset minimum, the max is the mean value plus one standard deviation.

The initializer only takes two arguments:

1. `webmap_item`: The `arcgis.gis.Item` object of the webmap to update.
1. `gis`: The `arcgis.gis.GIS` object of the AGOL organization.

Methods

- `update_color_ramp_values(self, layer_name, column_name, stops=5)`
  - Updates the color ramp stop values for `column_name` of `layer_name` in `self.webmap_item` using `stops` number of stops. Returns a boolean indicating success (passed through from `self.webmap_item.update()` call).
