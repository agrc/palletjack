from palletjack import GoogleDriveDownloader, GSheetLoader

sheet_id = ''

gsheetloader = GSheetLoader(r'c:\gis\git\pj-uorg\src\uorg\sheets-sa.json')
worksheets = gsheetloader.load_all_worksheets_into_dataframes(sheet_id)

uorg_2021 = worksheets['2021']
pics_list = list(uorg_2021['Picture'])

downloader = GoogleDriveDownloader(r'c:\temp\google_python_tests')
for pic in pics_list:
    if not pic:
        print('No link in cell')
        continue
    try:
        downloader.download_image_from_google_drive(pic)
    except RuntimeError as err:
        print(err)
