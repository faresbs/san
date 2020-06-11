"""
Standalone script to download and extract content
Use this to save and extract Pheonix dataset
"""

import wget
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

#Give it Url of file you want to download
#Path, where you want to save and extract
def download(zip_path, path, to_download=True):

	with ZipFile(wget.download(url, path), 'r') as zip_ref:
	   	zip_ref.extractall(path)
	

if len(sys.argv) < 3:
    print('Usage: download.py <url> <where to extract> ')
    sys.exit(0)

model_path = sys.argv[1]
label_path = sys.argv[2]