__author__ = 'redhat'

from pprint import pprint
from preprocess import load_obj
import os
from constants import *
from scipy import io as scio

filenames = [f for f in os.listdir(DIR) if os.path.isfile(DIR + f)]

for filename in filenames:
    print('\n\n' + filename)
    if '.pkl' in filename:
        filename_without_ext = filename[:-4]
        pprint(load_obj(filename_without_ext))
    elif '.mat' in filename:
        print(DIR + filename)
        pprint(scio.loadmat(DIR + filename))
