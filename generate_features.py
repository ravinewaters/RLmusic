__author__ = 'redhat'

from scipy.sparse import csr_matrix
from scipy import io
from common_methods import *


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
    if action == -1:
        return 99
    if state[-1] == -1:
        return 100
    if action[-1] == -1:
        # next fig is rest
        return 101
    if state[2] == 1:
        # pickup, no chord
        return 102
    chord_root = chords_dict[1][state[2]]
    next_chord_root = chords_dict[1][action[1]]
    int_chord_root = CHORD_ROOT_TO_INT[parse_chord(chord_root)]
    int_next_chord_root = CHORD_ROOT_TO_INT[parse_chord(next_chord_root)]
    return int_next_chord_root - int_chord_root


def get_bar_number(state):
    return state[0]


def get_current_figure_head_pitch(state):
    if state[-1] == -1:
        return 100
    return state[-1]


def compute_figure_head_movement(state, action):
    if action == -1:
        # exit action
        return 99
    if state[-1] == -1:
        # current is rest
        return 100
    if action[-1] == -1:
        # next fig is rest
        return 101
    return action[-1] - state[-1]

def get_current_beat(state):
    return state[3]


def get_next_beat(state):
    return state[4] + state[3]


def is_in_goal_state(state, action, term_states):
    if state in term_states and action == -1:
        return 1
    return 0


def is_rest(state):
    if state[-1] == -1:
        return 1
    return 0


def compute_features(state, action, fignotes_dict, chords_dict, term_states):
    # (root mvt,
    # current bar number
    # current fighead pitch,
    # fighead mvt,
    # get_current_beat
    # get_next_beat,
    # is_in_goal_state,
    # is_rest,
    # )

    feat_l = [
        compute_root_movement(state, action, chords_dict),
        get_bar_number(state),
        get_current_figure_head_pitch(state),
        compute_figure_head_movement(state, action),
        get_current_beat(state),
        get_next_beat(state),
        is_in_goal_state(state, action, term_states),
        is_rest(state),
    ]
    return feat_l

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
    # modify cols and rows list
    # called after we know the size of each coordinate

    index = 0
    for i in range(len(feat)):
        index = index + coord_size[i]
        cols.append(index + feat[i])
        rows.append(row_idx)

def generate_features_expectation_table():
    # assume original states
    fignotes_dict = load_obj('FIGNOTES_DICT')
    chords_dict = load_obj('CHORDS_DICT')
    term_states = load_obj('TERM_STATES')
    q_states = load_obj('Q_STATES')

    num_of_features = 8
    dictionaries = [{} for _ in range(num_of_features)]
    counters = [0] * num_of_features

    temp_dict = {}  # store proper features temporarily {row_idx: feat}

    # this loop is to get proper features and the size of each coordinate
    for state, actions in q_states.items():
        for action, value in actions.items():
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

    coord_size = [0] + [len(dict) for dict in dictionaries]
    print('coord_size:', coord_size[1:])

    # use row_idx as the row number to store feat_exp into
    # use csr_matrix
    cols = []  # this will be a list of column
    rows = []
    n_rows = -1
    for state, actions in q_states.items():
        for value in actions.values():
            row_idx = value[0]
            feat = temp_dict[row_idx]
            compute_binary_features_expectation(cols,
                                                rows,
                                                row_idx,
                                                feat,
                                                coord_size)
            if row_idx > n_rows:
                n_rows = row_idx

    # create csr_matrix from data, rows and cols.
    data = [1] * len(cols)
    n_rows += 1  # include index 0
    n_cols = sum(coord_size) + 1
    print('n_rows:', n_rows)
    print('n_cols:', n_cols)
    mtx = csr_matrix((data, (rows, cols)),
                     dtype=np.uint8,
                     shape=(n_rows, n_cols))

    # save matrix to file
    io.savemat(DIR + 'FEATURES_EXPECTATION_MATRIX', {'mtx': mtx})
    return mtx

if __name__ == "__main__":
    generate_features_expectation_table()