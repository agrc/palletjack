<!-- markdownlint-disable -->

<a href="..\transform#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `transform`
Transform pandas dataframes in preparation for loading to AGOL. 



---

<a href="..\transform\APIGeocoder#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `APIGeocoder`
Geocode a dataframe using the UGRC Web API Geocoder. 

Instantiate an APIGeocoder object with an api key from developer.mapserv.utah.gov. It will attempt to validate the API key. If validation fails, it will raise one of the following errors: 


- RuntimeError: If there was a network or other error 
- ValueError: If the key is invalid 
- UserWarning: If the API responds with some other abnormal result 

The individual geocoding steps are exposed in the `palletjack.utils.Geocoding` class in the utils module for use in other settings. 

<a href="..\transform\__init__#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(api_key)
```



**Args:**
 
 - <b>`api_key`</b> (str):  API key obtained from developer.mapserv.utah.gov 




---

<a href="..\transform\geocode_dataframe#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `geocode_dataframe`

```python
geocode_dataframe(
    dataframe,
    street_col,
    zone_col,
    wkid,
    rate_limits=(0.015, 0.03),
    **api_args
)
```

Geocode a pandas dataframe into a spatially-enabled dataframe 

Addresses that don't meet the threshold for geocoding (score > 70) are returned as points at 0,0 



**Args:**
 
 - <b>`dataframe`</b> (pd.DataFrame):  Input data with separate columns for street address and zip or city 
 - <b>`street_col`</b> (str):  The column containing the street address 
 - <b>`zone_col`</b> (str):  The column containing either the zip code or the city name 
 - <b>`wkid`</b> (int):  The projection to return the x/y points in 
 - <b>`rate_limits`</b> (Tuple <float>):  A lower and upper bound in seconds for pausing between API calls. Defaults to  (0.015, 0.03) 
 - <b>`**api_args (dict)`</b>:  Keyword arguments to be passed as parameters in the API GET call. 



**Returns:**
 
 - <b>`pd.DataFrame.spatial`</b>:  Geocoded data as a spatially-enabled DataFrame 


---

<a href="..\transform\FeatureServiceMerging#L99"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FeatureServiceMerging`
Get the live dataframe from a feature service and update it from another dataframe  






---

<a href="..\transform\get_live_dataframe#L139"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_live_dataframe`

```python
get_live_dataframe(gis, feature_service_itemid, layer_index=0)
```

Get a spatially-enabled dataframe representation of a hosted feature layer 



**Args:**
 
 - <b>`gis`</b> (arcgis.gis.GIS):  GIS object of the desired organization 
 - <b>`feature_service_itemid`</b> (str):  itemid in the gis of the desired hosted feature service 
 - <b>`layer_index`</b> (int, optional):  Index of the desired layer within the hosted feature service. Defaults to 0. 



**Raises:**
 
 - <b>`RuntimeError`</b>:  If it fails to load the data 



**Returns:**
 
 - <b>`pd.DataFrame.spatial`</b>:  Spatially-enabled dataframe representation of the hosted feature layer 

---

<a href="..\transform\update_live_data_with_new_data#L103"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update_live_data_with_new_data`

```python
update_live_data_with_new_data(live_dataframe, new_dataframe, join_column)
```

Update a dataframe with data from another 



**Args:**
 
 - <b>`live_dataframe`</b> (pd.DataFrame):  The dataframe containing info to be updated 
 - <b>`new_dataframe`</b> (pd.DataFrame):  Dataframe containing source info to use in the update 
 - <b>`join_column`</b> (str):  The column with unique IDs to be used as a key between the two dataframes 



**Raises:**
 
 - <b>`ValueError`</b>:  If the join_column is missing from either live or new data 
 - <b>`RuntimeWarning`</b>:  If there are rows in the new data that are not found in the live data; these will not be  added to the live dataframe. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  The updated dataframe, with data types converted via .convert_dtypes() 


---

<a href="..\transform\DataCleaning#L166"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DataCleaning`
Static methods for cleaning dataframes prior to uploading to AGOL  






---

<a href="..\transform\rename_dataframe_columns_for_agol#L240"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `rename_dataframe_columns_for_agol`

```python
rename_dataframe_columns_for_agol(dataframe)
```

Rename all the columns in a dataframe to valid AGOL column names 



**Args:**
 
 - <b>`dataframe`</b> (pd.DataFrame):  Dataframe to be renamed 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  Input dataframe with renamed columns 

---

<a href="..\transform\switch_to_datetime#L222"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `switch_to_datetime`

```python
switch_to_datetime(dataframe, date_fields, **to_datetime_kwargs)
```

Convert specified fields to datetime dtypes to ensure proper date formatting for AGOL 



**Args:**
 
 - <b>`dataframe`</b> (pd.DataFrame):  The source dataframe 
 - <b>`date_fields`</b> (List[int]):  The fields to convert to datetime 
 - <b>`**to_datetime_kwargs (keyword arguments, optional)`</b>:  Arguments to pass through to pd.to_datetime 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  The source dataframe with converted fields. 

---

<a href="..\transform\switch_to_float#L195"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `switch_to_float`

```python
switch_to_float(dataframe, fields_that_should_be_floats)
```

Convert specified fields to float, converting empty strings to None first as required 



**Args:**
 
 - <b>`dataframe`</b> (pd.DataFrame):  Input dataframe with columns to be converted 
 - <b>`fields_that_should_be_floats`</b> (list[str]):  List of column names to be converted 



**Raises:**
 
 - <b>`TypeError`</b>:  If any of the conversions fail. Often caused by values that aren't castable to floats  (non-empty, non-numeric strings, etc) 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  Input dataframe with columns converted to float 

---

<a href="..\transform\switch_to_nullable_int#L170"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `switch_to_nullable_int`

```python
switch_to_nullable_int(dataframe, fields_that_should_be_ints)
```

Convert specified fields to panda's nullable Int64 type to preserve int to EsriFieldTypeInteger mapping 



**Args:**
 
 - <b>`dataframe`</b> (pd.DataFrame):  Input dataframe with columns to be converted 
 - <b>`fields_that_should_be_ints`</b> (list[str]):  List of column names to be converted 



**Raises:**
 
 - <b>`TypeError`</b>:  If any of the conversions fail. Often caused by values that aren't int-castable floats (ie. x.0)  or np.nans. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  Input dataframe with columns converted to nullable Int64 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
