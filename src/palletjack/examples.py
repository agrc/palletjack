from pathlib import Path
from tempfile import TemporaryDirectory

import arcgis

from palletjack import (
    ColorRampReclassifier, FeatureServiceAttachmentsUpdater, FeatureServiceInlineUpdater, GoogleDriveDownloader,
    GSheetLoader, SFTPLoader
)


def load_google_sheet_then_download_and_update_attachments():
    """This example shows how you would download a google sheet as a dictionary of dataframes, then extract google drive sharing links from a column of one of the dataframes and use them to update the attachments of a hosted feature layer.
    """

    #: Settings and variables
    #: These would normally go in a secrets file and a config file depending upon exposure risk
    sheet_id = ''
    attachments_join_field = 'join_field'
    attachment_column = 'Picture'
    service_account_json = r'c:\foo\bar-sa.json'
    out_dir = r'c:\temp\google_python_tests'
    agol_org = 'https://utah.maps.arcgis.com'
    agol_user = ''
    agol_password = ''
    feature_layer_itemid = 'agol_item_id'

    #: Use a GSheetLoader to load a google sheet into a dictionary of dataframes
    gsheetloader = GSheetLoader(service_account_json)
    worksheets = gsheetloader.load_all_worksheets_into_dataframes(sheet_id)

    #: Use a GoogleDriveDownloader to download all the pictures from a single worksheet dataframe
    uorg_2021_dataframe = worksheets['2021']
    pics_list = list(uorg_2021_dataframe[attachment_column])

    downloader = GoogleDriveDownloader(out_dir)
    for pic_link in pics_list:
        #: if the link is an empty string or null, notify the user and don't try to download
        if not pic_link:
            print('No URL in attachment column')
            continue
        #: Skids are responsible for handling errors. In this case, if it can't access the link, print out the error
        try:
            downloader.download_image_from_google_drive(pic_link)
        except RuntimeError as err:
            print(err)

    #: Create an attachments dataframe by subsetting down to just the two fields and dropping any rows with null/empty attachments
    attachments_dataframe = uorg_2021_dataframe[[attachments_join_field, attachment_column]] \
                                               .copy().dropna(subset=attachment_column)
    #: Create the full path by prepending the output directory using .apply and a lambda function
    attachments_dataframe['full_file_path'] = attachments_dataframe[attachment_column].apply(
        lambda filename: str(Path(out_dir, filename))
    )

    #: General attachments dataframe layout:
    #:      | join_field | attachment_path_field |
    #: -----|------------|-----------------------|
    #:    0 |     1a     |  c:/foo/bar/baz.png   |
    #:    1 |     2b     |  c:/foo/bar/boo.png   |

    #: Get our GIS object via the ArcGIS API for Python
    gis = arcgis.gis.GIS(agol_org, agol_user, agol_password)

    #: Create our attachment updater and update attachments using the attachments dataframe
    attachment_updater = FeatureServiceAttachmentsUpdater(gis)
    attachment_updater.update_attachments(
        feature_layer_itemid, attachments_join_field, 'full_file_path', attachments_dataframe
    )


def download_from_sftp_update_agol_reclassify_map():
    """Download a csv from an sftp share. Use a join key from the csv to update rows in a feature service from the csv. Finally, reclassify the unclassed class breaks in the layer's renderer in a web map to reflect the new data.

    Condensed from https://github.com/agrc/erap-skid/, which is configured to run as a google cloud function.
    """

    #: Settings and variables
    #: These would normally go in a secrets file and a config file depending upon exposure risk
    local_knownhosts_path = 'path/to/knownhosts/file'
    sftp_host = 'sftp hostname or ip address'
    sftp_username = ''
    sftp_password = ''
    sftp_folder = 'folder/on/sftp/server'
    sftp_filename = 'project.csv'
    csv_datatypes = {
        'zip5': str,
        'Count_': str,
        'Amount': int,
        'Updated': str,
    }
    join_key_column = 'column_that_joins_csv_to_feature_service_data'

    agol_org = 'https://utah.maps.arcgis.com'
    agol_user = ''
    agol_password = ''
    feature_layer_itemid = 'agol_item_id'
    map_item_id = 'agol item id of webmap with layer to reclassify'
    map_layer_name = 'name of layer in map to reclassify'
    classification_field = 'field to use for layer reclassification'

    #: Set up a temp dir to hold the downloaded csv; be sure to clean up at the end
    tempdir = TemporaryDirectory()
    tempdir_path = Path(tempdir.name)

    #: Get our gis AGOL org object
    gis = arcgis.gis.GIS(agol_org, agol_user, agol_password)
    webmap_item = gis.content.get(map_item_id)  # pylint:disable=no-member

    #: Load the latest data from FTP
    sftp_loader = SFTPLoader(sftp_host, sftp_username, sftp_password, local_knownhosts_path, tempdir_path)
    number_of_files_downloaded = sftp_loader.download_sftp_folder_contents(sftp_folder=sftp_folder)
    dataframe = sftp_loader.read_csv_into_dataframe(sftp_filename, csv_datatypes)

    #: Update the AGOL data
    erap_updater = FeatureServiceInlineUpdater(gis, dataframe, join_key_column)
    number_of_rows_updated = erap_updater.update_existing_features_in_hosted_feature_layer(
        feature_layer_itemid, list(csv_datatypes.keys())
    )

    #: Reclassify the break values on the webmap's color ramp
    erap_reclassifier = ColorRampReclassifier(webmap_item, gis)
    success = erap_reclassifier.update_color_ramp_values(map_layer_name, classification_field)

    #: update returns either True or False to denote success or failure
    reclassifier_result = 'Success'
    if not success:
        reclassifier_result = 'Failure'

    #: Try to clean up the tempdir (we don't use a context manager); print any errors as a heads up
    try:
        tempdir.cleanup()
    except Exception as error:
        print(error)
