__author__ = 'redhat'

import pickle
import os

def make_flat_list(list_of_lists):
    flat_list = [item for lists in list_of_lists for item in lists]
    return flat_list

def save_obj(obj, name):
    dir = 'obj/'
    make_dir_when_not_exist(dir)
    with open(dir + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    dir = 'obj/'
    make_dir_when_not_exist(dir)
    with open(dir + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def make_dir_when_not_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)