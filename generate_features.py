__author__ = 'redhat'

# import numpy as np
# from constants import *
from scipy.sparse import csr_matrix
from scipy import io
from common_methods import *
# from itertools import product
# from pprint import pprint

# all wrong need to reconsider all these
def parse_chord(chord):
    if len(chord) == 1:
        return chord
    chord = chord[:2]
    if chord[1] == '#' or chord[1] == 'b':
        return chord[:2]
    else:
        return chord[0]


def compute_root_movement(state, action, chords_dict):
    # TROUBLE, how to parse chord?
    chord_root = chords_dict[1][state[1]]
    next_chord_root = chords_dict[1][action[1]]
    int_chord_root = CHORD_ROOT_TO_INT[parse_chord(chord_root)]
    int_next_chord_root = CHORD_ROOT_TO_INT[parse_chord(next_chord_root)]
    return int_next_chord_root - int_chord_root


def compute_figure_head_movement(state, action):
    return action[-1] - state[-1]


def get_fig_contour(state, fignotes_dict):
    #  UP or DOWN or STAY
    # state is an integer tuple
    fignotes = fignotes_dict[1][state[0]]
    fig_contour = fignotes[0] - fignotes[-2]
    if fig_contour > 0:
        return 2
    elif fig_contour < 0:
        return 0
    else:
        return 1


def compute_next_fig_beat(state):
    return (state[3] + state[2])/2


def is_in_term_state(state, term_states):
    if state in term_states:
        return 1
    return 0


def is_to_rest(action):
    if action[-1] == -1:
        return 1
    return 0


def compute_features(state, action, fignotes_dict, chords_dict,term_states):
    # (root mvt,
    # fighead mvt,
    # fig contour,
    # next_fig_contour,
    # to_term_state,
    # from_pickup,
    # to_rest,
    # from_rest,
    # next_beat,
    # )

    # from rest and not to rest
    if state[-1] == -1 and action[-1] != -1:
        tup = [
            0,
            0,
            0,
            get_fig_contour(action, fignotes_dict),
            is_in_term_state(state, term_states),
            0,
            0,
            1,
            compute_next_fig_beat(state),
        ]
    # to rest
    elif action[-1] == -1:
        # from rest
        if state[-1] == -1:
            tup = [
                0, 0, 0, 0, 0, 0, 1, 1, compute_next_fig_beat(state),
            ]
        # not from rest
        else:
            tup = [
                0, 0, 0, 0, 0, 0, 1, 0, compute_next_fig_beat(state),
            ]
    # pickup when chord == 1
    elif state[1] == 1:
        tup = [
            0,
            compute_figure_head_movement(state, action),
            get_fig_contour(state, fignotes_dict),
            get_fig_contour(action, fignotes_dict),
            is_in_term_state(state, term_states),
            1,
            is_to_rest(action),
            0,
            compute_next_fig_beat(state),
        ]
    else:
        tup = [
            compute_root_movement(state, action, chords_dict),
            compute_figure_head_movement(state, action),
            get_fig_contour(state, fignotes_dict),
            get_fig_contour(action, fignotes_dict),
            is_in_term_state(state, term_states),
            0,
            is_to_rest(action),
            0,
            compute_next_fig_beat(state),
        ]
    return tup

############


def compute_proper_features(state, action, fignotes_dict,
                            chords_dict, term_states, dictionaries, counters,
                            num_of_features):
    # use dictionary to get unique value of each coordinates so there is no
    # wasted coordinates in the binary features.
    feat = compute_features(state, action, fignotes_dict,
                           chords_dict, term_states)
    for i in range(num_of_features):
        if feat[i] not in dictionaries[i]:
            dictionaries[i][feat[i]] = counters[i]
            counters[i] += 1
        feat[i] = dictionaries[i][feat[i]]
    return feat


def compute_binary_features_expectation(cols, rows, row_idx, feat, coord_size):
    # Append to list cols
    # compute given state and action features expectation.
    # return number of columns which have value 1

    # wrong logic
    # length of each coordinate is changing every loop in the parent function.
    index = 0
    for i in range(len(feat)):
        index = index + coord_size[i]
        cols.append(index + feat[i])
        rows.append(row_idx)

def generate_features_expectation_table():
    fignotes_dict = load_obj('FIGNOTES_DICT')
    chords_dict = load_obj('CHORDS_DICT')
    term_states = load_obj('TERM_STATES')

    try:
        q_states = load_obj('Q_STATES')
        print('Q_STATES loaded.')
    except FileNotFoundError:
        all_states = load_obj('ALL_STATES')
        all_actions = load_obj('ALL_ACTIONS')
        list_of_all_states = [k+v for k, v in all_states.items()]
        list_of_all_actions = [k+v for k, v in all_actions.items()]
        q_states = generate_all_possible_q_states(list_of_all_states,
                                                  list_of_all_actions)

    num_of_features = 9
    dictionaries = [{} for x in range(num_of_features)]
    counters = [0] * num_of_features

    temp_dict = {}  # store proper features temporarily {row_idx: feat}

    # this loop is to get proper features and the size of each coordinate
    for state, actions in q_states.items():
        for action, value in actions.items():
            if action == - 1:
                continue
            row_idx = value[0]
            feat = compute_proper_features(state,
                                                action,
                                                fignotes_dict,
                                                chords_dict,
                                                term_states,
                                                dictionaries,
                                                counters,
                                                num_of_features)
            temp_dict[row_idx] = feat

    coord_size = [len(dict) for dict in dictionaries]
    coord_size = np.concatenate((np.array([0]), coord_size))
    print('coord_size:', coord_size)

    # use row_idx as the row number to store feat_exp into
    # use csr_matrix
    cols = []  # this will be a list of column
    rows = []
    counter = 1
    for state, actions in q_states.items():
        for action, value in actions.items():
            if action == -1:
                continue
            print('\nstate:', state)
            print('action:', action)
            row_idx = value[0]
            feat = temp_dict[row_idx]
            compute_binary_features_expectation(cols,
                                                rows,
                                                row_idx,
                                                feat,
                                                coord_size)
            counter += 1

    # create csr_matrix from data, rows and cols.
    data = [1] * len(cols)
    n_rows = counter
    n_cols = sum(coord_size + 1)
    mtx = csr_matrix((data, (rows, cols)),
                     dtype=np.uint8,
                     shape=(n_rows, n_cols))

    # save matrix to file
    io.savemat(DIR + 'FEATURES_EXPECTATION_MATRIX', {'mtx': mtx})
    return mtx

if __name__ == "__main__":
    generate_features_expectation_table()