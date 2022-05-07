from .common_utils import *
import os, shutil


def auto_mkdir(path):

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
               
            elif os.path.isdir(file_path):
                #print("---------")
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))