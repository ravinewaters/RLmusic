__author__ = 'redhat'

import pickle
from random import random
import bisect
import os
from constants import DIR


def make_flat_list(list_of_lists):
    flat_list = [item for lists in list_of_lists for item in lists]
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


def compute_next_state(state, action):
    bar = state[0]
    beat = state[3] + state[4]
    if beat == 20:
        beat = 4
        bar += 1
    s_prime = (bar,
               action[0], action[1],
               beat,
               action[2],
               action[3])
    return s_prime


def weighted_choice_b(weights):
    totals = []
    running_total = 0
    
    for w in weights:
        running_total += w
        totals.append(running_total)

    rnd = random() * running_total
    return bisect.bisect_right(totals, rnd)

if __name__ == '__main__':
    pass
