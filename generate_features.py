__author__ = 'redhat'

import scipy
from constants import *

def map_tup_to_bin_tup(tup, coord_size):
    """
    Assume that range of element in each coordinate is range(0, size)
    len(tup) = len(coord_size)
    convert (1,1,1) to (1, 0, 1, 0, 1, 0) given that the first coord has
    size 2, second size 2 and third size 3
    Big-Endian
    """

    bin_tup = [0] * sum(coord_size)
    coord_size = (0,) + coord_size
    pos = 0
    for i in range(len(tup)):
        pos = pos + coord_size[i]
        bin_tup[pos + tup[i]] = 1
    return bin_tup


def compute_next_state(state, action):
    # states_dict: (states_to_int, int_to_states)
    # actions_dict: (actions_to_int, int_to_actions)
    # a = ((next_fig_seq_of_notes), next_chord_name)
    # s = ((fig_seq_of_notes), chord_name, duration, beat, fighead_note)
    # s_prime = ((next_fig_seq_of_notes),
    # next_chord_name,
    # duration = sum of duration of each note in next_fig_seq_of_notes,
    # beat = duration of s + beat s in,
    # fighead_note = the first note of next_fig_seq_of_notes)

    s_prime = (action[0], action[1],
               sum(action[0][1::2]),
               state[2]+state[3],
               action[0][0])
    return s_prime

# These methods are assumed to have input the original state and action not
# integers.
def compute_root_movement(state, action):
    current_chord = CHORD_ROOT_TO_INT[state[1]]
    next_chord = CHORD_ROOT_TO_INT[action[1]]
    return next_chord - current_chord

def compute_figure_head_movement(state, action):
    return action[0][0] - state[-1]

def get_fig_contour(state):
    "UP or DOWN or STAY"
    fig_contour = state[-2] - state[0]
    if fig_contour > 0:
        return 1
    elif fig_contour < 0 :
        return -1
    else:
        return 0

def compute_next_fig_beat(state):
    return state[2] + state[3]

def is_to_term_state(action):
    if compute_next_state(state, action) in TERM_STATES:
        return 1
    return 0

def is_from_pickup(state):
    if state[1] == 'pickup':
        return 1
    return 0

def is_to_rest(action):
    if action[1] == 'rest':
        return 1
    return 0