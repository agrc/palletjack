<!-- markdownlint-disable -->

<a href="..\load#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `load`
Modify existing ArcGIS Online content (mostly hosted feature services). Contains classes for updating hosted feature service data, modifying the attachments on a hosted feature service, or modifying map symbology. 



---

<a href="..\load\FeatureServiceUpdater#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FeatureServiceUpdater`
Update an AGOL Feature Service with data from a pandas DataFrame. 

Contains four class methods that can be called directly without needing to instantiate an object: add_features, remove_features, update_features, and truncate_and_load_features. 

Because the update process uploads the data as geojson, all input geometries must be in WGS84 (wkid 4326). Input dataframes can be projected using dataframe.spatial.project(4326). ArcGIS Online will then project the uploaded data to match the hosted feature service's projection. 

<a href="..\load\__init__#L164"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    gis,
    feature_service_itemid,
    dataframe=None,
    fields=None,
    failsafe_dir=None,
    layer_index=0
)
```








---

<a href="..\load\add_features#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `add_features`

```python
add_features(gis, feature_service_itemid, dataframe, layer_index=0)
```

Adds new features to existing hosted feature layer from new dataframe. 

The new dataframe must have a 'SHAPE' column containing geometries of the same type as the live data. The dataframe must have a WGS84 (wkid 4326) projection. New OBJECTIDs will be automatically generated. 

The new dataframe's columns and data must match the existing data's fields (with the exception of generated fields like shape area and length) in name, type, and allowable length. Live fields that are not nullable and don't have a default value must have a value in the new data; missing data in these fields will raise an error. 



**Args:**
 
 - <b>`gis`</b> (arcgis.gis.GIS):  GIS item for AGOL org 
 - <b>`features_service_item_id`</b> (str):  itemid for service to update 
 - <b>`dataframe`</b> (pd.DataFrame.spatial):  Spatially enabled dataframe of data to be added 
 - <b>`layer_index`</b> (int):  Index of layer within service to update. Defaults to 0. 



**Raises:**
 
 - <b>`ValueError`</b>:  If the new field and existing fields don't match, the SHAPE field is missing or has an  incompatible type, the new data contains null fields, the new data exceeds the existing field  lengths, or a specified field is missing from either new or live data. 



**Returns:**
 
 - <b>`int`</b>:  Number of features added 

---

<a href="..\load\remove_features#L61"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `remove_features`

```python
remove_features(gis, feature_service_itemid, delete_oids, layer_index=0)
```

Deletes features from a hosted feature layer based on comma-separated string of Object IDs 

This is a wrapper around the arcgis.FeatureLayer.delete_features method that adds some sanity checking. The delete operation is rolled-back if any of the features fail to delete using (rollback_on_failure=True). This function will raise a RuntimeError as well after delete_features() returns if any of them fail. 

The sanity checks will raise errors or warnings as appropriate if any of them fail. 



**Args:**
 
 - <b>`delete_oids`</b> (list[int]):  List of OIDs to delete 



**Raises:**
 
 - <b>`ValueError`</b>:  If delete_string can't be split on `,` 
 - <b>`TypeError`</b>:  If any of the items in delete_string can't be cast to ints 
 - <b>`ValueError`</b>:  If delete_string is empty 
 - <b>`UserWarning`</b>:  If any of the Object IDs in delete_string don't exist in the live data 
 - <b>`RuntimeError`</b>:  If any of the OIDs fail to delete 



**Returns:**
 
 - <b>`int`</b>:  The number of features deleted 

---

<a href="..\load\truncate_and_load_features#L127"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `truncate_and_load_features`

```python
truncate_and_load_features(
    gis,
    feature_service_itemid,
    dataframe,
    failsafe_dir='',
    layer_index=0
)
```

Overwrite a hosted feature layer by truncating and loading the new data 

When the existing dataset is truncated, a copy is kept in memory as a spatially-enabled dataframe. If the new data fail to load, this copy is reloaded. If the reload fails, the copy is written to failsafe_dir with the filename {todays_date}.json (2022-12-31.json). 

The new dataframe must have a 'SHAPE' column containing geometries of the same type as the live data. The dataframe must have a WGS84 (wkid 4326) projection. New OBJECTIDs will be automatically generated. 

The new dataframe's columns and data must match the existing data's fields (with the exception of generated fields like shape area and length) in name, type, and allowable length. Live fields that are not nullable and don't have a default value must have a value in the new data; missing data in these fields will raise an error. 



**Args:**
 
 - <b>`gis`</b> (arcgis.gis.GIS):  GIS item for AGOL org 
 - <b>`feature_service_itemid`</b> (str):  itemid for service to update 
 - <b>`dataframe`</b> (pd.DataFrame.spatial):  Spatially enabled dataframe of new data to be loaded 
 - <b>`failsafe_dir`</b> (str, optional):  Directory to save original data in case of complete failure. If left blank, existing data won't be saved. Defaults to '' 
 - <b>`layer_index`</b> (int, optional):  Index of layer within service to update. Defaults to 0. 



**Returns:**
 
 - <b>`int`</b>:  Number of features loaded 

---

<a href="..\load\update_features#L88"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `update_features`

```python
update_features(
    gis,
    feature_service_itemid,
    dataframe,
    layer_index=0,
    update_geometry=True
)
```

Updates existing features within a hosted feature layer using OBJECTID as the join field. 

The new data can have either attributes and geometries or only attributes based on the update_geometry flag. A combination of updates from a source with both attributes & geometries and a source with attributes-only must be done with two separate calls. The geometries must be provided in a SHAPE column, be the same type as the live data, and have a WGS84 (wkid 4326) projection. 

The new dataframe's columns and data must match the existing data's fields (with the exception of generated fields like shape area and length) in name, type, and allowable length. Live fields that are not nullable and don't have a default value must have a value in the new data; missing data in these fields will raise an error. 



**Args:**
 
 - <b>`gis`</b> (arcgis.gis.GIS):  GIS item for AGOL org 
 - <b>`features_service_item_id`</b> (str):  itemid for service to update 
 - <b>`dataframe`</b> (pd.DataFrame.spatial):  Spatially enabled dataframe of data to be updated 
 - <b>`layer_index`</b> (int):  Index of layer within service to update. Defaults to 0. 
 - <b>`update_geometry`</b> (bool):  Whether to update attributes and geometry (True) or just attributes (False).  Defaults to False. 



**Raises:**
 
 - <b>`ValueError`</b>:  If the new field and existing fields don't match, the SHAPE field is missing or has an  incompatible type, the new data contains null fields, the new data exceeds the existing field  lengths, or a specified field is missing from either new or live data. 



**Returns:**
 
 - <b>`int`</b>:  Number of features updated 


---

<a href="..\load\FeatureServiceAttachmentsUpdater#L418"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FeatureServiceAttachmentsUpdater`
Add or overwrite attachments in a feature service using a dataframe of the desired "new" attachments. 

Updates the attachments based on a dataframe containing two columns: a join key present in the live data (the dataframe column name must match the feature service field name) and the path of the file to attach to the feature. While AGOL supports multiple attachments, this only accepts a single file as input. 

If a matching feature in AGOl doesn't have an attachment, the file referred to by the dataframe will be uploaded. If it does have an attachment, it checks the existing filename with the referenced file. If they are different, the file from the dataframe will be updated. If they are the same, nothing happens. 

<a href="..\load\__init__#L430"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(gis)
```



**Args:**
 
 - <b>`gis`</b> (arcgis.gis.GIS):  The AGOL organization's gis object 




---

<a href="..\load\build_attachments_dataframe#L657"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `build_attachments_dataframe`

```python
build_attachments_dataframe(
    input_dataframe,
    join_column,
    attachment_column,
    out_dir
)
```

Create an attachments dataframe by subsetting down to just the two fields and dropping any rows  with null/empty attachments 



**Args:**
 
 - <b>`input_dataframe`</b> (pd.DataFrame):  Input data containing at least the join and attachment filename columns 
 - <b>`join_column`</b> (str):  Unique key joining attachments to live data 
 - <b>`attachment_column`</b> (str):  Filename for each attachment 
 - <b>`out_dir`</b> (str or Path):  Output directory, will be used to build full path to attachment 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  Dataframe with join key, attachment name, and full attachment paths 

---

<a href="..\load\update_attachments#L610"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update_attachments`

```python
update_attachments(
    feature_layer_itemid,
    attachment_join_field,
    attachment_path_field,
    attachments_df,
    layer_number=0
)
```

Update a feature layer's attachments based on info from a dataframe of desired attachment file names 

Depends on a dataframe populated with a join key for the live data and the downloaded or locally-available attachments. If the name of the "new" attachment is the same as an existing attachment for that feature, it is not updated. If it is different or there isn't an existing attachment, the "new" attachment is attached to that feature. 



**Args:**
 
 - <b>`feature_layer_itemid`</b> (str):  The AGOL Item ID of the feature layer to update 
 - <b>`attachment_join_field`</b> (str):  The field containing the join key between the attachments dataframe and the  live data 
 - <b>`attachment_path_field`</b> (str):  The field containing the desired attachment file path 
 - <b>`attachments_df`</b> (pd.DataFrame):  A dataframe of desired attachments, including a join key and the local path  to the attachment 
 - <b>`layer_number`</b> (int, optional):  The layer within the Item ID to update. Defaults to 0. 



**Returns:**
 
 - <b>`(int, int)`</b>:  Tuple of counts of successful overwrites and adds. 


---

<a href="..\load\ColorRampReclassifier#L682"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ColorRampReclassifier`
Updates the interval ranges on a webmap's layer's classification renderer based on the layer's current data. 

Manually edits the JSON definition to change a layer's color ramp values based on a simple unclassed scheme similar to AGOL's unclassed ramp. The minimum value is the dataset minimum, the max is the mean value plus one standard deviation. 

<a href="..\load\__init__#L690"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(webmap_item, gis)
```



**Args:**
 
 - <b>`webmap_item`</b> (arcgis.mapping.WebMap):  The webmap item in the AGOL organization 
 - <b>`gis`</b> (arcgis.gis.GIS):  The AGOL organization as a gis object 




---

<a href="..\load\update_color_ramp_values#L794"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update_color_ramp_values`

```python
update_color_ramp_values(layer_name, column_name, stops=5)
```

Update the color ramp ranges for layer_name in self.webmap_item. 

Does not alter colors or introduce additional stops; only overwrites the values for existing breaks. 



**Args:**
 
 - <b>`layer_name`</b> (str):  The exact name of the layer to be updated 
 - <b>`column_name`</b> (str):  The name of the attribute being displayed as an (un)classified range 
 - <b>`stops`</b> (int, optional):  The number of stops to calculate. Must match existing stops. Defaults to 5. 



**Returns:**
 
 - <b>`Bool`</b>:  Success or failure of update operation 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
