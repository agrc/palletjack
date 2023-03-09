<!-- markdownlint-disable -->

# API Overview

## Modules

- [`extract`](./extract.md#module-extract): Extract tabular/spatial data from various sources into a pandas dataframe.
- [`load`](./load.md#module-load): Modify existing ArcGIS Online content (mostly hosted feature services). Contains classes for updating hosted feature
- [`transform`](./transform.md#module-transform): Transform pandas dataframes in preparation for loading to AGOL.
- [`utils`](./utils.md#module-utils): Utility classes and methods that are used internally throughout palletjack. Many are exposed publicly in case they are useful elsewhere in a client's code.

## Classes

- [`extract.GSheetLoader`](./extract.md#class-gsheetloader): Loads data from a Google Sheets spreadsheet into a pandas data frame
- [`extract.GoogleDriveDownloader`](./extract.md#class-googledrivedownloader): Provides methods to download any non-html file (ie, Content-Type != text/html) Google Drive file from it's
- [`extract.PostgresLoader`](./extract.md#class-postgresloader): Loads data from a Postgres/PostGIS database into a pandas data frame
- [`extract.SFTPLoader`](./extract.md#class-sftploader): Loads data from an SFTP share into a pandas DataFrame
- [`load.ColorRampReclassifier`](./load.md#class-colorrampreclassifier): Updates the interval ranges on a webmap's layer's classification renderer based on the layer's current data.
- [`load.FeatureServiceAttachmentsUpdater`](./load.md#class-featureserviceattachmentsupdater): Add or overwrite attachments in a feature service using a dataframe of the desired "new" attachments.
- [`load.FeatureServiceUpdater`](./load.md#class-featureserviceupdater): Update an AGOL Feature Service with data from a pandas DataFrame.
- [`transform.APIGeocoder`](./transform.md#class-apigeocoder): Geocode a dataframe using the UGRC Web API Geocoder.
- [`transform.DataCleaning`](./transform.md#class-datacleaning): Static methods for cleaning dataframes prior to uploading to AGOL
- [`transform.FeatureServiceMerging`](./transform.md#class-featureservicemerging): Get the live dataframe from a feature service and update it from another dataframe
- [`utils.Chunking`](./utils.md#class-chunking): Divide a dataframe into chunks to satisfy upload size requirements for append operation.
- [`utils.DeleteUtils`](./utils.md#class-deleteutils): Verify Object IDs used for delete operations
- [`utils.FieldChecker`](./utils.md#class-fieldchecker): Check the fields of a new dataframe against live data. Each method will raise errors if its checks fail.
- [`utils.Geocoding`](./utils.md#class-geocoding): Methods for geocoding an address

## Functions

- [`utils.authorize_pygsheets`](./utils.md#function-authorize_pygsheets): Authenticate pygsheets using either a service file or google.auth.credentials.Credentials object.
- [`utils.build_sql_in_list`](./utils.md#function-build_sql_in_list): Generate a properly formatted list to be a target for a SQL 'IN' clause
- [`utils.calc_modulus_for_reporting_interval`](./utils.md#function-calc_modulus_for_reporting_interval): Calculate a number that can be used as a modulus for splitting n up into 10 or 20 intervals, depending on
- [`utils.check_field_set_to_unique`](./utils.md#function-check_field_set_to_unique): Makes sure field_name has a "unique constraint" in AGOL, which allows it to be used for .append upserts
- [`utils.check_fields_match`](./utils.md#function-check_fields_match): Make sure new data doesn't have any extra fields, warn if it doesn't contain all live fields
- [`utils.check_index_column_in_feature_layer`](./utils.md#function-check_index_column_in_feature_layer): Ensure index_column is present for any future operations
- [`utils.fix_numeric_empty_strings`](./utils.md#function-fix_numeric_empty_strings): Replace empty strings with None for numeric fields that allow nulls
- [`utils.get_null_geometries`](./utils.md#function-get_null_geometries): Generate placeholder geometries near 0, 0 with type based on provided feature layer properties dictionary.
- [`utils.rename_columns_for_agol`](./utils.md#function-rename_columns_for_agol): Replace special characters and spaces with '_' to match AGOL field names
- [`utils.rename_fields`](./utils.md#function-rename_fields): Rename fields based on field_mapping
- [`utils.retry`](./utils.md#function-retry): Allows you to retry a function/method three times to overcome network jitters
- [`utils.save_feature_layer_to_json`](./utils.md#function-save_feature_layer_to_json): Save a feature_layer to directory for safety as {layer name}_{todays date}.json


---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
