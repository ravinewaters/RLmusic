__author__ = 'redhat'

from scipy import sparse
import numpy as np
from constants import *
from preprocess import load_obj
from pprint import pprint

def tuple_to_int(tup, elem_size):
    array = np.array(tup)
    sizes = np.array((1,) + elem_size[:-1])
    cum_prod = np.cumprod(sizes)
    return np.dot(array, cum_prod)

def is_valid_action(state, action):
    # valid action iff
    # current_beat + duration + action duration <= 5.0
    fig_duration = state[2]
    fig_beat = state[3]
    action_fig_duration = sum(action[0][1::2])
    if fig_beat + fig_duration + action_fig_duration <= 5.0:
        return True
    return False

def map_tup_to_bin_array(tup, min_elem, max_elem):
    """
    !!!!!!!!!!! Only need the position not the tuple !!!!!!!!!!!!
    Each coordinate range is specified in min_elem and max_elem
    len(tup) = len(min_elem) = len(max_elem)
    convert (1,1,1) to (1, 0, 1, 0, 1, 0) given that the first coord has
    size 2, second size 2 and third size 3
    Big-Endian
    """
    coord_size = np.array(max_elem) - np.array(min_elem) + 1
    bin_array_length = sum(coord_size)
    coord_size = np.concatenate((np.array([0]), coord_size))
    pos = []
    index = 0
    for i in range(len(tup)):
        index = index + coord_size[i]
        pos.append(index + tup[i] - min_elem[i])
    return np.array(pos), bin_array_length

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

def get_features_range(states_dict, actions_dict, terminal_states):
    for state in states_dict[0]:
        if state in terminal_states:
            continue
        for action in actions_dict[0]:
            if not is_valid_action(state, action):
                continue

            features_vector = compute_features(state, action, terminal_states)

            try:
                min_features_vector = np.minimum(min_features_vector, features_vector)
                max_features_vector = np.maximum(max_features_vector, features_vector)
            except NameError:
                min_features_vector = features_vector
                max_features_vector = features_vector

    return (min_features_vector, max_features_vector)

# These methods are assumed to have input the original state and action not
# integers.

def parse_chord(chord):
    pass

def compute_root_movement(state, action):
    # TROUBLE, how to parse chord?
    current_chord = CHORD_ROOT_TO_INT[state[1][0]]
    next_chord = CHORD_ROOT_TO_INT[action[1][0]]
    return next_chord - current_chord

def compute_figure_head_movement(state, action):
    return action[0][0] - state[-1]

def get_fig_contour(state):
    "UP or DOWN or STAY"
    fig_contour = state[0][-2] - state[0][0]
    if fig_contour > 0:
        return 1
    elif fig_contour < 0 :
        return -1
    else:
        return 0

def compute_next_fig_beat(state):
    return 2*(state[2] + state[3])

def is_to_term_state(state, action, TERM_STATES):
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

def compute_features(state, action, term_states):
    if state[1] == 'rest' or action[1] == 'rest':
        tup = (0, 0, 0, 0, 0, 1)
    else:
        tup = (
            compute_root_movement(state, action),
            compute_figure_head_movement(state, action),
            get_fig_contour(state),
            compute_next_fig_beat(state),
            is_to_term_state(state, action, term_states),
            is_to_rest(action),
        )
    return np.array(tup)

############

def generate_features(states_dict, actions_dict, terminal_states):
    state_action_sizes = (len(states_dict[0]), len(actions_dict[0]))  #(# states, # actions)
    min_feat, max_feat = get_features_range(states_dict, actions_dict,
                                     terminal_states)
    num_of_rows = state_action_sizes[0] * state_action_sizes[1]

    # Use DOK sparse matrix

    first = True
    for state in states_dict[0]:
        if state in terminal_states:
            continue
        for action in actions_dict[0]:
            if not is_valid_action(state, action):
                continue
            int_s = states_dict[0][state]
            int_a = actions_dict[0][action]
            features_vector = compute_features(state, action, terminal_states)

            row = tuple_to_int((int_s, int_a), state_action_sizes)
            if first:
                # when first iteration, initialize sparse matrix after
                # having computer number of columns of features vector
                col, num_of_cols = map_tup_to_bin_array(features_vector,
                                                        min_feat,
                                                        max_feat)
                sparse_feature_matrix = sparse.dok_matrix((num_of_rows,
                                                           num_of_cols),
                                                          dtype=np.uint8)
                first = False
            else:
                col, _ = map_tup_to_bin_array(features_vector, min_feat, max_feat)
            for j in col:
                sparse_feature_matrix[row, j] = 1

    return sparse_feature_matrix

if __name__ == "__main__":
    STATES_DICT = load_obj('STATES_DICT')
    ACTIONS_DICT = load_obj('ACTIONS_DICT')
    TERM_STATES = load_obj('TERM_STATES')
    pprint(generate_features(STATES_DICT, ACTIONS_DICT, TERM_STATES))
