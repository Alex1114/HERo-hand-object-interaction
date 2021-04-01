import os
import sys
import logging
import gdown
from zipfile import ZipFile


dataset_url = 'https://drive.google.com/u/1/uc?id=1q6lZBLIOk1c3jOfEJYDDbqMa5uSoSgR0'
dataset_name = 'dataset'
if not os.path.isdir(dataset_name):
    gdown.download(dataset_url, output=dataset_name + '.zip', quiet=False)
    zip1 = ZipFile(dataset_name + '.zip')
    zip1.extractall()
    zip1.close()
 

print("Finished downloading dataset.")