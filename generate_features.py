__author__ = 'redhat'

from scipy import sparse, io
import numpy as np
from constants import *
from common_methods import *

# need state_action_dict otherwise slow.


def map_tup_to_bin_array(tup, min_elem, max_elem):
    """
    Goal: turns a tuple into a binary array.
    Input: a tuple of integers, the min (and max) value each coordinate
    can have.
    Output: array of indices that correspond to value 1 in binary tuple of
    the original tuple. Also output the length of the binary tuple

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


def get_features_range(states_dict, actions_dict, terminal_states):
    for state in states_dict[0]:
        if state in terminal_states:
            continue
        for action in actions_dict[0]:
            if not is_valid_action(state, action):
                continue

            features_vector = compute_features(state, action, terminal_states)

            try:
                min_features_vector = np.minimum(min_features_vector,
                                                 features_vector)
                max_features_vector = np.maximum(max_features_vector,
                                                 features_vector)
            except NameError:
                min_features_vector = features_vector
                max_features_vector = features_vector

    return min_features_vector, max_features_vector


# These methods are assumed to have input the original state and action not
# integers.

def parse_chord(chord):
    chord = chord[:2]
    if chord[1] == '#' or chord[1] == 'b':
        return chord[:2]
    else:
        return chord[0]


def compute_root_movement(state, action):
    # TROUBLE, how to parse chord?
    int_chord_root = CHORD_ROOT_TO_INT[parse_chord(state[1])]
    int_next_chord_root = CHORD_ROOT_TO_INT[parse_chord(action[1])]
    return int_next_chord_root - int_chord_root


def compute_figure_head_movement(state, action):
    return action[0][0] - state[-1]


def get_fig_contour(state):
    #  UP or DOWN or STAY
    fig_contour = state[0][-2] - state[0][0]
    if fig_contour > 0:
        return 1
    elif fig_contour < 0:
        return -1
    else:
        return 0


def compute_next_fig_beat(int_state):
    return (int_state[2] + int_state[3])


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

def map_state_action_pair(states_dict, actions_dict):
    state_action_to_int_dict = {}
    int_to_state_action_dict = {}
    state_action_sizes = (len(states_dict[0]), len(actions_dict[0]))
    for int_s in states_dict[1]:
        for int_a in actions_dict[1]:
            if is_valid_action(states_dict[1][int_s], actions_dict[1][int_a]):
                integer = array_to_int((int_s, int_a), state_action_sizes)
                state_action_to_int_dict[int_s, int_a] = integer
                int_to_state_action_dict[integer] = (int_s, int_a)
    return state_action_to_int_dict, int_to_state_action_dict

def generate_features_matrix(elem_sizes, fignotes_dict, chords_dict,
                             terminal_states):
    state_action_sizes = (
        len(states_dict[0]), len(actions_dict[0]))  # (# states, # actions)
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

            # row = array_to_int((int_s, int_a), state_action_sizes)
            row = state_action_dict[0][int_s, int_a]
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
                col, _ = map_tup_to_bin_array(features_vector, min_feat,
                                              max_feat)
            for j in col:
                sparse_feature_matrix[row, j] = 1

    return sparse_feature_matrix.tocsr()

if __name__ == "__main__":
    states_dict = load_obj('STATES_DICT')
    actions_dict = load_obj('ACTIONS_DICT')
    term_states = load_obj('TERM_STATES')
    state_action_dict = map_state_action_pair(states_dict, actions_dict)
    save_obj(state_action_dict, 'STATE_ACTION_DICT')
    features_matrix = generate_features_matrix(states_dict, actions_dict,
                                               term_states, state_action_dict)
    io.savemat(DIR + 'FEATURES_MATRIX.mat',
               {'features_matrix': features_matrix})
