<!-- markdownlint-disable -->

<a href="..\utils#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `utils`
Utility classes and methods that are used internally throughout palletjack. Many are exposed publicly in case they are useful elsewhere in a client's code. 


---

<a href="..\utils\retry#L25"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `retry`

```python
retry(worker_method, *args, **kwargs)
```

Allows you to retry a function/method three times to overcome network jitters 

Retries worker_method three times (for a total of four tries, including the initial attempt), pausing 2^trycount seconds between each retry. Any arguments for worker_method can be passed in as additional parameters to retry() following worker_method: retry(foo_method, arg1, arg2, keyword_arg=3) 



**Args:**
 
 - <b>`worker_method`</b> (callable):  The name of the method to be retried (minus the calling parens) 



**Raises:**
 
 - <b>`error`</b>:  The final error that causes worker_method to fail after 3 retries 



**Returns:**
 
 - <b>`various`</b>:  The value(s) returned by worked_method 


---

<a href="..\utils\rename_columns_for_agol#L68"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rename_columns_for_agol`

```python
rename_columns_for_agol(columns)
```

Replace special characters and spaces with '_' to match AGOL field names 



**Args:**
 
 - <b>`columns`</b> (iter):  The new columns to be renamed 



**Returns:**
 
 - <b>`Dict`</b>:  Mapping {'original name': 'cleaned_name'} 


---

<a href="..\utils\check_fields_match#L91"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `check_fields_match`

```python
check_fields_match(featurelayer, new_dataframe)
```

Make sure new data doesn't have any extra fields, warn if it doesn't contain all live fields 



**Args:**
 
 - <b>`featurelayer`</b> (arcgis.features.FeatureLayer):  Live data 
 - <b>`new_dataframe`</b> (pd.DataFrame):  New data 



**Raises:**
 
 - <b>`RuntimeError`</b>:  If new data contains a field not present in the live data 


---

<a href="..\utils\check_index_column_in_feature_layer#L122"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `check_index_column_in_feature_layer`

```python
check_index_column_in_feature_layer(featurelayer, index_column)
```

Ensure index_column is present for any future operations 



**Args:**
 
 - <b>`featurelayer`</b> (arcgis.features.FeatureLayer):  The live feature layer 
 - <b>`index_column`</b> (str):  The index column meant to link new and live data 



**Raises:**
 
 - <b>`RuntimeError`</b>:  If index_column is not in featurelayer's fields 


---

<a href="..\utils\rename_fields#L139"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `rename_fields`

```python
rename_fields(dataframe, field_mapping)
```

Rename fields based on field_mapping 



**Args:**
 
 - <b>`dataframe`</b> (pd.DataFrame):  Dataframe with columns to be renamed 
 - <b>`field_mapping`</b> (dict):  Mapping of existing field names to new names 



**Raises:**
 
 - <b>`ValueError`</b>:  If an existing name from field_mapping is not found in dataframe.columns 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  Dataframe with renamed fields 


---

<a href="..\utils\build_sql_in_list#L163"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `build_sql_in_list`

```python
build_sql_in_list(series)
```

Generate a properly formatted list to be a target for a SQL 'IN' clause 



**Args:**
 
 - <b>`series`</b> (pd.Series):  Series of values to be included in the 'IN' list 



**Returns:**
 
 - <b>`str`</b>:  Values formatted as (1, 2, 3) for numbers or ('a', 'b', 'c') for anything else 


---

<a href="..\utils\check_field_set_to_unique#L180"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `check_field_set_to_unique`

```python
check_field_set_to_unique(featurelayer, field_name)
```

Makes sure field_name has a "unique constraint" in AGOL, which allows it to be used for .append upserts 



**Args:**
 
 - <b>`featurelayer`</b> (arcgis.features.FeatureLayer):  The target feature layer 
 - <b>`field_name`</b> (str):  The AGOL-valid field name to check 



**Raises:**
 
 - <b>`RuntimeError`</b>:  If the field is not unique (or if it's indexed but not unique) 


---

<a href="..\utils\calc_modulus_for_reporting_interval#L314"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `calc_modulus_for_reporting_interval`

```python
calc_modulus_for_reporting_interval(n, split_value=500)
```

Calculate a number that can be used as a modulus for splitting n up into 10 or 20 intervals, depending on split_value. 



**Args:**
 
 - <b>`n`</b> (int):  The number to divide into intervals 
 - <b>`split_value`</b> (int, optional):  The point at which it should create 20 intervals instead of 10. Defaults to 500. 



**Returns:**
 
 - <b>`int`</b>:  Number to be used as modulus to compare to 0 in reporting code 


---

<a href="..\utils\authorize_pygsheets#L335"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `authorize_pygsheets`

```python
authorize_pygsheets(credentials)
```

Authenticate pygsheets using either a service file or google.auth.credentials.Credentials object. 

Requires either the path to a service account .json file that has access to the files in question or  a `google. auth.credentials.Credentials` object. Calling `google.auth.default()` in a Google Cloud Function will give you a tuple of a `Credentials` object and the project id. You can use this `Credentials` object to authorize pygsheets as the same account the Cloud Function is running under. 

Tries first to load credentials from file; if this fails tries credentials directly as a custom_credential. 



**Args:**
 
 - <b>`credentials`</b> (str or google.auth.credentials.Credentials):  Path to the service file OR credentials object  obtained from google.auth.default() within a cloud function. 



**Raises:**
 
 - <b>`RuntimeError`</b>:  If both authorization method attempts fail 



**Returns:**
 
 - <b>`pygsheets.Client`</b>:  Authorized pygsheets client 


---

<a href="..\utils\save_feature_layer_to_json#L367"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `save_feature_layer_to_json`

```python
save_feature_layer_to_json(feature_layer, directory)
```

Save a feature_layer to directory for safety as {layer name}_{todays date}.json 



**Args:**
 
 - <b>`feature_layer`</b> (arcgis.features.FeatureLayer):  The FeatureLayer object to save to disk. 
 - <b>`directory`</b> (str or Path):  The directory to save the data to. 



**Returns:**
 
 - <b>`Path`</b>:  The full path to the output file, named with the layer name and today's date. 


---

<a href="..\utils\get_null_geometries#L687"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_null_geometries`

```python
get_null_geometries(feature_layer_properties)
```

Generate placeholder geometries near 0, 0 with type based on provided feature layer properties dictionary. 



**Args:**
 
 - <b>`feature_layer_properties`</b> (dict):  .properties from a feature layer item, contains 'geometryType' key 



**Raises:**
 
 - <b>`NotImplementedError`</b>:  If we get a geometryType we haven't implemented a null-geometry generator for 



**Returns:**
 
 - <b>`arcgis.geometry.Geometry`</b>:  A geometry object of the corresponding type centered around null island. 


---

<a href="..\utils\fix_numeric_empty_strings#L912"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fix_numeric_empty_strings`

```python
fix_numeric_empty_strings(feature_set, feature_layer_fields)
```

Replace empty strings with None for numeric fields that allow nulls 

**Args:**
 
 - <b>`feature_set`</b> (arcgis.features.FeatureSet):  Feature set to clean 
 - <b>`fields`</b> (Dict):  fields from feature layer 


---

<a href="..\utils\Geocoding#L200"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Geocoding`
Methods for geocoding an address  






---

<a href="..\utils\geocode_addr#L204"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `geocode_addr`

```python
geocode_addr(street, zone, api_key, rate_limits, **api_args)
```

Geocode an address through the UGRC Web API geocoder 

Invalid results are returned with an x,y of 0,0, a score of 0.0, and a match address of 'No Match' 



**Args:**
 
 - <b>`street`</b> (str):  The street address 
 - <b>`zone`</b> (str):  The zip code or city 
 - <b>`api_key`</b> (str):  API key obtained from developer.mapserv.utah.gov 
 - <b>`rate_limits`</b> (Tuple <float>):  A lower and upper bound in seconds for pausing between API calls. Defaults to  (0.015, 0.03) 
 - <b>`**api_args (dict)`</b>:  Keyword arguments to be passed as parameters in the API GET call. The API key will be  added to this dict. 



**Returns:**
 
 - <b>`tuple[int]`</b>:  The match's x coordinate, y coordinate, score, and match address 

---

<a href="..\utils\validate_api_key#L284"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `validate_api_key`

```python
validate_api_key(api_key)
```

Check to see if a Web API key is valid by geocoding a single, known address point 



**Args:**
 
 - <b>`api_key`</b> (str):  API Key 



**Raises:**
 
 - <b>`RuntimeError`</b>:  If there was a network or other error attempting to geocode the known point 
 - <b>`ValueError`</b>:  If the API responds with an invalid key message 
 - <b>`UserWarning`</b>:  If the API responds with some other abnormal result 


---

<a href="..\utils\FieldChecker#L391"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FieldChecker`
Check the fields of a new dataframe against live data. Each method will raise errors if its checks fail. Provides the check_fields class method to run all the checks in one call with having to create an object. 

<a href="..\utils\__init__#L418"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(live_data_properties, new_dataframe)
```



**Args:**
 
 - <b>`live_data_properties`</b> (dict):  FeatureLayer.properties of live data 
 - <b>`new_dataframe`</b> (pd.DataFrame):  New data to be checked 




---

<a href="..\utils\check_field_length#L577"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_field_length`

```python
check_field_length(fields)
```

Raise an error if a new data string value is longer than allowed in the live data. 



**Args:**
 
 - <b>`fields`</b> (List[str]):  Fields to check 



**Raises:**
 
 - <b>`ValueError`</b>:  If the string fields in the new data contain a value longer than the corresponding field in the  live data allows. 

---

<a href="..\utils\check_fields#L396"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `check_fields`

```python
check_fields(live_data_properties, new_dataframe, fields, add_oid)
```

Run all the field checks, raising errors and warnings where they fail. 

Check individual method docstrings for details and specific errors raised. 



**Args:**
 
 - <b>`live_data_properties`</b> (dict):  FeatureLayer.properties of live data 
 - <b>`new_dataframe`</b> (pd.DataFrame):  New data to be checked 
 - <b>`fields`</b> (List[str]):  Fields to check 
 - <b>`add_oid`</b> (bool):  Add OBJECTID to fields if its not already present (for operations that are dependent on  OBJECTID, such as upsert) 

---

<a href="..\utils\check_fields_present#L606"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_fields_present`

```python
check_fields_present(fields, add_oid)
```

Raise an error if the fields to be operated on aren't present in either the live or new data. 



**Args:**
 
 - <b>`fields`</b> (List[str]):  The fields to be operated on. 
 - <b>`add_oid`</b> (bool):  Add OBJECTID to fields if its not already present (for operations that are dependent on  OBJECTID, such as upsert) 



**Raises:**
 
 - <b>`RuntimeError`</b>:  If any of fields are not in live or new data. 

---

<a href="..\utils\check_for_non_null_fields#L541"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_for_non_null_fields`

```python
check_for_non_null_fields(fields)
```

Raise an error if the new data contains nulls in a field that the live data says is not nullable. 

If this error occurs, the client should use pandas fillna() method to replace NaNs/Nones with empty strings or appropriate nodata values. 



**Args:**
 
 - <b>`fields`</b> (List[str]):  Fields to check 



**Raises:**
 
 - <b>`ValueError`</b>:  If the new data contains nulls in a field that the live data says is not nullable and doesn't  have a default value. 

---

<a href="..\utils\check_live_and_new_field_types_match#L429"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_live_and_new_field_types_match`

```python
check_live_and_new_field_types_match(fields)
```

Raise an error if the field types of the live and new data don't match. 

Uses a dictionary mapping Esri field types to pandas dtypes. If 'SHAPE' is included in the fields, it calls _check_geometry_types to verify the spatial types are compatible. 



**Args:**
 
 - <b>`fields`</b> (List[str]):  Fields to be updated 



**Raises:**
 
 - <b>`ValueError`</b>:  If the field types or spatial types are incompatible, the new data has multiple geometry types,  or the new data is not a valid spatially-enabled dataframe. 
 - <b>`NotImplementedError`</b>:  If the live data has a field that has not yet been mapped to a pandas dtype. 

---

<a href="..\utils\check_nullable_ints_shapely#L660"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_nullable_ints_shapely`

```python
check_nullable_ints_shapely()
```

Raise a warning if null values occur within nullable integer fields of the dataframe 

Apparently due to a convention within shapely, any null values in an integer field are converted to 0. 



**Raises:**
 
 - <b>`UserWarning`</b>:  If we're using shapely instead of arcpy, the new dataframe uses nullable int dtypes, and there  is one or more pd.NA values within a nullable int column. 

---

<a href="..\utils\check_srs_wgs84#L636"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_srs_wgs84`

```python
check_srs_wgs84()
```

Raise an error if the new spatial reference system isn't WGS84 as required by geojson. 



**Raises:**
 
 - <b>`ValueError`</b>:  If the new SRS value can't be cast to an int (please log an issue if this occurs) 
 - <b>`ValueError`</b>:  If the new SRS value isn't 4326. 


---

<a href="..\utils\DeleteUtils#L732"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DeleteUtils`
Verify Object IDs used for delete operations  






---

<a href="..\utils\check_delete_oids_are_in_live_data#L776"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_delete_oids_are_in_live_data`

```python
check_delete_oids_are_in_live_data(oid_string, numeric_oids, feature_layer)
```

Warn if a delete Object ID doesn't exist in the live data, return number missing 



**Args:**
 
 - <b>`oid_string`</b> (str):  Comma-separated string of delete Object IDs 
 - <b>`numeric_oids`</b> (list[int]):  The parsed and cast-to-int Object IDs 
 - <b>`feature_layer`</b> (arcgis.features.FeatureLayer):  Live FeatureLayer item 



**Raises:**
 
 - <b>`UserWarning`</b>:  If any of the Object IDs in numeric_oids don't exist in the live data. 



**Returns:**
 
 - <b>`int`</b>:  Number of Object IDs missing from live data 

---

<a href="..\utils\check_delete_oids_are_ints#L736"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_delete_oids_are_ints`

```python
check_delete_oids_are_ints(oid_list)
```

Raise an error if a list of strings can't be parsed as ints 



**Args:**
 
 - <b>`oid_list`</b> (list[int]):  List of Object IDs to delete 



**Raises:**
 
 - <b>`TypeError`</b>:  If any of the items in oid_list can't be cast to ints 



**Returns:**
 
 - <b>`list[int]`</b>:  oid_list converted to ints 

---

<a href="..\utils\check_for_empty_oid_list#L761"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_for_empty_oid_list`

```python
check_for_empty_oid_list(oid_list, numeric_oids)
```

Raise an error if the parsed Object ID list is empty 



**Args:**
 
 - <b>`oid_list`</b> (list[int]):  The original list of Object IDs to delete 
 - <b>`numeric_oids`</b> (list[int]):  The cast-to-int Object IDs 



**Raises:**
 
 - <b>`ValueError`</b>:  If numeric_oids is empty 


---

<a href="..\utils\Chunking#L802"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Chunking`
Divide a dataframe into chunks to satisfy upload size requirements for append operation.  






---

<a href="..\utils\build_upload_json#L844"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `build_upload_json`

```python
build_upload_json(dataframe, feature_layer_fields, max_bytes=100000000)
```

Create list of geojson strings of spatially-enabled DataFrame, divided into chunks if it exceeds max_bytes 

Recursively chunks dataframe to ensure no one chunk is larger than max_bytes. Converts all empty strings in nullable numeric fields in feature sets created from individual chunks to None prior to converting to geojson to ensure the field stays numeric. 



**Args:**
 
 - <b>`dataframe`</b> (pd.DataFrame.spatial):  Spatially-enabled dataframe to be converted to geojson 
 - <b>`feature_layer_fields`</b>:  All the fields from the feature layer (feature_layer.properties.fields) 
 - <b>`max_bytes`</b> (int, optional):  Maximum size in bytes any one geojson string can be. Defaults to 100000000 (AGOL  text uploads are limited to 100 MB?) 



**Returns:**
 
 - <b>`list[str]`</b>:  A list of the dataframe chunks converted to geojson 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
