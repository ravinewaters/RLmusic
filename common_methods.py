__author__ = 'redhat'

import pickle
from random import random
import bisect
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
    elif fig_beat + fig_duration == 20:
        return True
    assert False, "Shouldn't get here"


def generate_all_possible_q_states(all_states, all_actions):
    # assume complete states and actions, not reduced ones.
    # initalize a dictionary

    # row_idx is a row number in which we store feat_exp of corresponding
    # state, action into.
    term_states = load_obj('TERM_STATES')
    row_idx = 0
    q_states = {}
    for state in all_states:
        if state in term_states:
            # key 'state' wasn't existed.
            # for exit action
            q_states[state] = {-1: 0}
        for action in all_actions:
            if action == -1:
                continue
            if is_valid_action(state, action):
                next_state = compute_next_state(state, action)
                if state in q_states:
                    q_states[state][action] = (row_idx, next_state)
                else:
                    q_states[state] = {action: (row_idx, next_state)}
                row_idx += 1
    save_obj(q_states, 'Q_STATES')
    return q_states


def weighted_choice(choices):
    choices = tuple(choices)
    total = sum(w for c, w in choices)
    r = random() * total
    upto = 0
    for c, w in choices:
      if upto + w >= r:
         return c
      upto += w
    assert False, "Shouldn't get here"
    

def weighted_choice_b(weights):
    totals = []
    running_total = 0
    
    for w in weights:
        running_total += w
        totals.append(running_total)

    rnd = random() * running_total
    return bisect.bisect_right(totals, rnd)

if __name__ == '__main__':
    all_states = load_obj('ALL_STATES')
    all_actions = load_obj('ALL_ACTIONS')
    list_of_all_states = [k+v for k, v in all_states.items()]
    list_of_all_actions = [k+v for k, v in all_actions.items()]
    q_states = generate_all_possible_q_states(list_of_all_states,
                                                  list_of_all_actions)
