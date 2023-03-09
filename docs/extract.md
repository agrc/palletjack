<!-- markdownlint-disable -->

<a href="..\extract#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `extract`
Extract tabular/spatial data from various sources into a pandas dataframe. 

Each different type of source has its own class. Each class may have multiple methods available for different loading operations or techniques. 



---

<a href="..\extract\GSheetLoader#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `GSheetLoader`
Loads data from a Google Sheets spreadsheet into a pandas data frame 

Requires either the path to a service account .json file that has access to the sheet in question or a `google.auth.credentials.Credentials` object. Calling `google.auth.default()` in a Google Cloud Function will give you a tuple of a `Credentials` object and the project id. You can use this `Credentials` object to authorize pygsheets as the same account the Cloud Function is running under. 

<a href="..\extract\__init__#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(credentials)
```



**Args:**
 
 - <b>`credentials`</b> (str or google.auth.credentials.Credentials):  Path to the service file OR credentials object  obtained from google.auth.default() within a cloud function. 




---

<a href="..\extract\combine_worksheets_into_single_dataframe#L84"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `combine_worksheets_into_single_dataframe`

```python
combine_worksheets_into_single_dataframe(worksheet_dfs)
```

Merge worksheet dataframes (having same columns) into a single dataframe with a new 'worksheet' column identifying the source worksheet. 



**Args:**
 
 - <b>`worksheet_dfs`</b> (dict):  {'worksheet_name': Worksheet as a dataframe}. 



**Raises:**
 
 - <b>`ValueError`</b>:  If all the worksheets in worksheets_dfs don't have the same column index, it raises an error  and bombs out. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  A single combined data frame with a new 'worksheet' column identifying the worksheet the row  came from. The row index is the original row numbers and is probably not unique. 

---

<a href="..\extract\load_all_worksheets_into_dataframes#L67"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_all_worksheets_into_dataframes`

```python
load_all_worksheets_into_dataframes(sheet_id)
```

Load all worksheets into a dictionary of dataframes. Keys are the worksheet. 



**Args:**
 
 - <b>`sheet_id`</b> (str):  The ID of the sheet (long alpha-numeric unique ID) 



**Returns:**
 
 - <b>`dict`</b>:  {'worksheet_name': Worksheet as a dataframe} 

---

<a href="..\extract\load_specific_worksheet_into_dataframe#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_specific_worksheet_into_dataframe`

```python
load_specific_worksheet_into_dataframe(sheet_id, worksheet, by_title=False)
```

Load a single worksheet from a spreadsheet into a dataframe by worksheet index or title 



**Args:**
 
 - <b>`sheet_id`</b> (str):  The ID of the sheet (long alpha-numeric unique ID) 
 - <b>`worksheet`</b> (int or str):  Zero-based index of the worksheet or the worksheet title 
 - <b>`by_title`</b> (bool, optional):  Search for worksheet by title instead of index. Defaults to False. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  The specified worksheet as a data frame. 


---

<a href="..\extract\GoogleDriveDownloader#L111"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `GoogleDriveDownloader`
Provides methods to download any non-html file (ie, Content-Type != text/html) Google Drive file from it's sharing link (of the form `https://drive.google.com/file/d/big_long_id/etc`). The files may be publicly shared or shared with a service account. 

This class has two similar sets of methods. The `*_using_api` methods authenticate to the Google API using either a service account file or a `google.auth.credentials.Credentials` object and downloads using the API. The `*_using_api` methods are the most robust and should be used whenever possible. 

<a href="..\extract\__init__#L120"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(out_dir)
```



**Args:**
 
 - <b>`out_dir`</b> (str or Path):  Directory to save downloaded files. Can be reassigned later to change the directory. 




---

<a href="..\extract\download_attachments_from_dataframe#L335"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `download_attachments_from_dataframe`

```python
download_attachments_from_dataframe(
    dataframe,
    sharing_link_column,
    join_id_column,
    output_path_column
)
```

Download the attachments linked in a dataframe column, creating a new column with the resulting path 



**Args:**
 
 - <b>`dataframe`</b> (pd.DataFrame):  Input dataframe with required columns 
 - <b>`sharing_link_column`</b> (str):  Column holding the Google sharing link 
 - <b>`join_id_column`</b> (str):  Column holding a unique key (for reporting purposes) 
 - <b>`output_path_column`</b> (str):  Column for the resulting path; will be added if it doesn't exist in the  dataframe 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  Input dataframe with output path info 

---

<a href="..\extract\download_attachments_from_dataframe_using_api#L354"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `download_attachments_from_dataframe_using_api`

```python
download_attachments_from_dataframe_using_api(
    credentials,
    dataframe,
    sharing_link_column,
    join_id_column,
    output_path_column
)
```

Download the attachments linked in a dataframe column using an authenticated api client, creating a new column with the resulting path 



**Args:**
 
 - <b>`credentials`</b> (str or google.auth.credentials.Credentials):  Path to the service file OR credentials object  obtained from google.auth.default() within a cloud function. 
 - <b>`dataframe`</b> (pd.DataFrame):  Input dataframe with required columns 
 - <b>`sharing_link_column`</b> (str):  Column holding the Google sharing link 
 - <b>`join_id_column`</b> (str):  Column holding a unique key (for reporting purposes) 
 - <b>`output_path_column`</b> (str):  Column for the resulting path; will be added if it doesn't existing in the  dataframe 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  Input dataframe with output path info 

---

<a href="..\extract\download_file_from_google_drive#L218"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `download_file_from_google_drive`

```python
download_file_from_google_drive(sharing_link, join_id, pause=0.0)
```

Download a publicly-shared image from Google Drive using it's sharing link 

Uses an anonymous HTTP request with support for sleeping between downloads to try to get around Google's blocking (I haven't found a good value yet). 

Logs a warning if the URL doesn't match the proper pattern or it can't extract the unique id from the sharing URL. Will also log a warning if the header's Content-Type is text/html, which usually indicates the HTTP response was an error message instead of the file. 



**Args:**
 
 - <b>`sharing_link`</b> (str):  The publicly-shared link to the image. 
 - <b>`join_id`</b> (str or int):  The unique key for the row (used for reporting) 
 - <b>`pause`</b> (flt, optional):  Pause the specified number of seconds before downloading. Defaults to 0. 



**Returns:**
 
 - <b>`Path`</b>:  Path of downloaded file or None if download fails/is not possible 

---

<a href="..\extract\download_file_from_google_drive_using_api#L299"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `download_file_from_google_drive_using_api`

```python
download_file_from_google_drive_using_api(gsheets_client, sharing_link, join_id)
```

Download a file using the Google API via pygsheets authentication. 

Requires a pygsheets client object that handles authentication. 

Logs a warning if the URL doesn't match the proper pattern or it can't extract the unique id from the sharing URL. Will also log a warning if the header's Content-Type is text/html, which usually indicates the HTTP response was an error message instead of the file. 



**Args:**
 
 - <b>`gsheets_client`</b> (pygsheets.Client):  The authenticated client object from pygsheets 
 - <b>`sharing_link`</b> (str):  Sharing link to the file to be downloaded 
 - <b>`join_id`</b> (str or int):  Unique key for the row (used for reporting) 



**Returns:**
 
 - <b>`Path`</b>:  Path of downloaded file or None if download fails/is not possible 


---

<a href="..\extract\SFTPLoader#L382"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SFTPLoader`
Loads data from an SFTP share into a pandas DataFrame  



<a href="..\extract\__init__#L386"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(host, username, password, knownhosts_file, download_dir)
```



**Args:**
 
 - <b>`host`</b> (str):  The SFTP host to connect to 
 - <b>`username`</b> (str):  SFTP username 
 - <b>`password`</b> (str):  SFTP password 
 - <b>`knownhosts_file`</b> (str):  Path to a known_hosts file for pysftp.CnOpts. Can be generated via ssh-keyscan. 
 - <b>`download_dir`</b> (str or Path):  Directory to save downloaded files 




---

<a href="..\extract\download_sftp_folder_contents#L403"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `download_sftp_folder_contents`

```python
download_sftp_folder_contents(sftp_folder='upload')
```

Download all files in sftp_folder to the SFTPLoader's download_dir 



**Args:**
 
 - <b>`sftp_folder`</b> (str, optional):  Path of remote folder, relative to sftp home directory. Defaults to 'upload'. 

---

<a href="..\extract\download_sftp_single_file#L426"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `download_sftp_single_file`

```python
download_sftp_single_file(filename, sftp_folder='upload')
```

Download filename into SFTPLoader's download_dir 



**Args:**
 
 - <b>`filename`</b> (str):  Filename to download; used as output filename as well. 
 - <b>`sftp_folder`</b> (str, optional):  Path of remote folder, relative to sftp home directory. Defaults to 'upload'. 



**Raises:**
 
 - <b>`FileNotFoundError`</b>:  Will warn if pysftp can't find the file or folder on the sftp server 



**Returns:**
 
 - <b>`Path`</b>:  Downloaded file's path 

---

<a href="..\extract\read_csv_into_dataframe#L459"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `read_csv_into_dataframe`

```python
read_csv_into_dataframe(filename, column_types=None)
```

Read filename into a dataframe with optional column names and types 



**Args:**
 
 - <b>`filename`</b> (str):  Name of file in the SFTPLoader's download_dir 
 - <b>`column_types`</b> (dict, optional):  Column names and their dtypes(np.float64, str, etc). Defaults to None. 



**Returns:**
 
 - <b>`pd.DataFrame`</b>:  CSV as a pandas dataframe 


---

<a href="..\extract\PostgresLoader#L482"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PostgresLoader`
Loads data from a Postgres/PostGIS database into a pandas data frame 

<a href="..\extract\__init__#L485"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(host, database, username, password, port=5432)
```



**Args:**
 
 - <b>`host`</b> (str):  Postgres server host name 
 - <b>`database`</b> (str):  Database to connect to 
 - <b>`username`</b> (str):  Database user 
 - <b>`password`</b> (str):  Database password 
 - <b>`port`</b> (int, optional):  Database port. Defaults to 5432. 




---

<a href="..\extract\read_table_into_dataframe#L520"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `read_table_into_dataframe`

```python
read_table_into_dataframe(table_name, index_column, crs, spatial_column)
```

Read a table into a dataframe 



**Args:**
 
 - <b>`table_name`</b> (str):  Name of table or view to read in the following format: schema.table_name 
 - <b>`index_column`</b> (str):  Name of column to use as the dataframe's index 
 - <b>`crs`</b> (str):  Coordinate reference system of the table's geometry column 
 - <b>`spatial_column`</b> (str):  Name of the table's geometry or geography column 



**Returns:**
 
 - <b>`pd.DataFrame.spatial`</b>:  Table as a spatially enabled dataframe 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
