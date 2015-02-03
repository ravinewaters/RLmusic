__author__ = 'redhat'

import pickle
import os
import numpy as np
from constants import *

def make_flat_list(list_of_lists):
    flat_list = (item for lists in list_of_lists for item in lists)
    return flat_list

def save_obj(obj, name):
    make_dir_when_not_exist(DIR)
    with open(DIR + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    make_dir_when_not_exist(DIR)
    with open(DIR + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def make_dir_when_not_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def array_to_int(arr, elem_size):
    arr = np.array(arr)
    sizes = np.array((1,) + elem_size[:-1])
    cum_prod = np.cumprod(sizes)
    return np.dot(arr, cum_prod)

def int_to_array(integer, elem_size):
    sizes = np.array((1,) + elem_size[:-1])
    cum_prod = np.cumprod(sizes)
    index = -1
    arr = [0]*len(elem_size)
    for radix in reversed(cum_prod):
        q, integer = divmod(integer, radix)
        arr[index] = q
        index -= 1
    return arr

def compute_next_state(state, action):
    beat = state[2] + state[3]
    if beat == 20:
        beat = 4
    s_prime = (action[0], action[1],
               beat,
               action[2],
               action[3])
    return s_prime

def is_valid_action(state, action):
    # valid action iff
    # current_beat + duration + action duration <= 20
    fig_duration = state[3]
    fig_beat = state[2]
    action_fig_duration = action[2]
    if fig_beat + fig_duration < 20:
        if action_fig_duration + fig_beat + fig_duration <= 20:
            return True
        return False # if > 20, false
    return True