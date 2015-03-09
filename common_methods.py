__author__ = 'redhat'

import pickle
import os
from constants import DIR


def save_obj(obj, name):
    make_dir_when_not_exist(DIR)
    with open(DIR + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    make_dir_when_not_exist(DIR)
    with open(DIR + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def make_dir_when_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


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

if __name__ == '__main__':
    pass
