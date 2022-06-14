from palletjack import GoogleDriveDownloader

out_dir = r'c:/temp/google_python_tests/'
downloader = GoogleDriveDownloader(out_dir)
outfile = downloader.download_file_from_google_drive(
    'https://drive.google.com/file/d/1YtJhQMcPSd2udmxlKnxEyeeYtUkAkSxt/view?usp=sharing', 'test'
)
