# Overview

`palletjack` is a library of modules for updating data in ArcGIS Online (AGOL) based on information from external data. It handles the repetitive parts of the extract, transform, and load stages of the ETL process, allowing the user to focus on the custom transform steps unique to every project.

As a library, `palletjack` is not meant to be used by itself. Rather, its classes and methods should be used by other apps written for specific use cases. These other apps are referred to as the "client" apps, and internally we often refer to them as "skids".

pandas dataframes are the main unifying data structure between the different steps. The client loads data from an external source into a dataframe and then modifies the dataframe according to their business needs. Once the dataframe is ready to go, it is used to update the hosted feature service in AGOL.

The API reference can be found [here](./api.md).

## Organization

The individual modules within `palletjack` each handle their own step of the ETL process. Each module contains classes for accomplishing its task organized by source, operation, or destination. There may be multiple, similar methods in a class depending on exactly how you want to perform a given step—you probably won't use all the available classes and methods in every application. The publicly-exposed methods usually call several private methods to keep functions small and testable.

Classes in `extract` handle the extract stage, pulling data in from external sources. Tabular data (csvs, Google Sheets) are loaded into dataframes, while non-tabular data for attachments are just downloaded to the specified locations. You'll instantiate the desired class with the basic connection info and then call the appropriate method on the resulting object to extract the data.

There are a handful of classes in `transform` with methods for cleaning and preparing your dataframes for upload to AGOL. You may also need to modify your data to fit your specific business needs: calculating fields, renaming fields, performing quality checks, etc. Some classes only have static methods can be called directly without needing to instantiate the class.

Once your dataframe is looking pretty, the `load` module will help you update a hosted feature service with your new data. The `FeatureServiceUpdater` class contains several class methods that handle the instantiation process for you, allowing you to make a single method call. The other classes in `load` require you to instantiate the class yourself.

While many parts of the classes' functionality are hidden in private methods, commonly-used code is exposed publicly in the `utils` module. You will probably not need any of the methods provided, but they may be useful for other projects. This is palletjack's junk drawer.

## Data Considerations

Under the hood, `palletjack` uses the `arcgis.features.FeatureLayer.append()` method to upload data. To eliminate the dependency on `arcpy` (and thus ArcGIS Pro/Enterprise), it converts and uploads the data as a geojson. This conversion process introduces several constraints and gotchas on the format of the data. `palletjack` tests for all the known gotchas and raises an error if the data needs extra work before uploading. In addition, AGOL imposes its own set of constraints.

### Field Names

The column names in your dataframes should match the field names in AGOL one-to-one, with the exception of AGOL's auto-generated fields (shape length/area, editor tracking, etc). You can use `transform.DataCleaning.rename_dataframe_columns_for_agol()` to handle the majority of field name formatting, but you'll want to double check the results.

### Field Types

The upload process is very particular about data types and missing data. `utils.FieldChecker.check_live_and_new_field_types_match()` contains a mapping of dataframe dtypes to Esri field types. They generally follow what you would expect. However, because pandas (currently) handles missing data by with `np.nan` by default, you may have integer data assigned a float dtype. In addition, some sources render missing data as an empty string, creating an object dtype. Finally, datetimes must be in UTC and stored in the non-timezone-aware `datetime64[ns]` dtype.

`transform.DataCleaning` has methods to help convert your data to these dtypes. In addition, it's a good practice to use pandas' nullable dtypes via [`pd.DataFrame.convert_dtypes()`](https://pandas.pydata.org/pandas-docs/dev/reference/api/pandas.DataFrame.convert_dtypes.html) (see also the section on [nullable ints](https://pandas.pydata.org/pandas-docs/dev/user_guide/integer_na.html)).

### Geometries

`palletjack` uses Esri's spatially-enabled dataframes for handling geometries. However, there's no reason you couldn't use geodataframes for other parts of the process and convert them to spatially-enabled dataframes for use in the `load` methods.

Because the upload process uses geojsons, you **MUST** project your dataframe to WGS84 (wkid 4326) (the upload stage will error out if it isn't). If your hosted feature service is in a different projection, AGOL will automatically project it back as part of the upload process.

**WARNING**: The reprojection process may introduce spatial shifts/topological errors due to projection differences. We highly recommend your target hosted feature services use WGS84 from the beginning, and that you choose an appropriate transformation when projecting your data.

### OBJECTID and Join Keys

If you want to update existing data without truncating and loading, you will need a join key between the incoming new data and the existing AGOL data. Do not use OBJECTID for this field; it may change at any time. Instead, use your own custom field that you have complete control over. You will perform the join manually in the transform step with pandas by loading the live AGOL data into a dataframe.

## Error handling

The client is responsible for handling errors and warnings that arise during the process. `palletjack` will raise its own errors when something occurs that would keep the procces from continuing or when one of its internal data checks fails. It will also captured and [chain](https://docs.python.org/3/tutorial/errors.html#exception-chaining) errors from the underlying libraries to include additional, context-specific messages for the user. It will raise warnings when something happens that the client should be aware of but will not keep the process from completing.

## Logging

`palletjack` takes full advantage of python's built-in [`logging` library](https://docs.python.org/3/howto/logging.html#advanced-logging-tutorial) to perform its feedback and reporting. It creates a main `palletjack` logger, and then each module and each class create their own child logger under this, becoming `palletjack.extract.SFTPLoader`, etc.

### Accessing `palletjack` logs

The client can get a reference to the `palletjack` logger and add their handlers, formatters, etc to it alongside its own logger:

```python
myapp_logger = logging.getLogger('my_app')
myapp_logger.setLevel(logging.INFO)
palletjack_logger = logging.getLogger('palletjack')
palletjack_logger.setLevel(logging.INFO)
#: set up handlers and formatters
#: ...
myapp_logger.addHandler(log_handler)
palletjack_logger.addHandler(log_handler)
```

### Log levels

- `logging.DEBUG` includes verbose debug info that should allow you to manually (possibly programmatically) undo or redo any operation.
- `logging.INFO` includes standard runtime progress reports and result information.
- `logging.WARNING` includes negative results from checks or other situations that the user should be aware of.
