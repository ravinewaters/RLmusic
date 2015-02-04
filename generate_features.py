__author__ = 'redhat'

import numpy as np
from constants import *
from scipy import sparse
from common_methods import *
# from pprint import pprint


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
    return state[3] + state[2]


def is_to_term_state(state, action, term_states):
    if compute_next_state(state, action) in term_states:
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
        tup = (
            0,
            0,
            0,
            get_fig_contour(action, fignotes_dict),
            is_to_term_state(state, action, term_states),
            0,
            0,
            1,
            compute_next_fig_beat(state),
        )
    # to rest
    elif action[-1] == -1:
        # from rest
        if state[-1] == -1:
            tup = (
                0, 0, 0, 0, 0, 0, 1, 1, compute_next_fig_beat(state),
            )
        # not from rest
        else:
            tup = (
                0, 0, 0, 0, 0, 0, 1, 0, compute_next_fig_beat(state),
            )
    # pickup when chord == 1
    elif state[1] == 1:
        tup = (
            0,
            compute_figure_head_movement(state, action),
            get_fig_contour(state, fignotes_dict),
            get_fig_contour(action, fignotes_dict),
            is_to_term_state(state, action, term_states),
            1,
            is_to_rest(action),
            0,
            compute_next_fig_beat(state),
        )
    else:
        tup = (
            compute_root_movement(state, action, chords_dict),
            compute_figure_head_movement(state, action),
            get_fig_contour(state, fignotes_dict),
            get_fig_contour(action, fignotes_dict),
            is_to_term_state(state, action, term_states),
            0,
            is_to_rest(action),
            0,
            compute_next_fig_beat(state),
        )
    return np.array(tup)

############

def compute_binary_features_expectation(state, action, min_elem, max_elem,
                                        fignotes_dict, chords_dict,
                                        term_states):
    # USE CSR_MATRIX
    # compute given state and action features expectation.
    tup = compute_features(state, action, fignotes_dict,
                           chords_dict, term_states)
    # print(tup)
    coord_size = np.array(max_elem) - np.array(min_elem) + 1
    coord_size = np.concatenate((np.array([0]), coord_size))
    col = []
    index = 0
    for i in range(len(tup)):
        index = index + coord_size[i]
        col.append(index + tup[i] - min_elem[i])
    data = [1]* len(col)
    row = [0] * len(col)
    mtx = sparse.csr_matrix((data, (row, col)), (1, sum(coord_size)))
    return mtx


if __name__ == "__main__":
    fignotes_dict = load_obj('FIGNOTES_DICT')
    chords_dict = load_obj('CHORDS_DICT')
    term_states = load_obj('TERM_STATES')
    all_actions = load_obj('ALL_ACTIONS')
    all_states = load_obj('ALL_STATES')
    elem_range = load_obj('ELEM_RANGE')


    # feat_exp = []
    # for states in all_states:
    #     for state in states:
    #         for action in all_actions:
    #             if state in term_states:
    #                 continue
    #             if not is_valid_action(state, action):
    #                 continue
    #             res = compute_binary_features_expectation(state, action, min_elem,
    #                                                       max_elem,
    #                                                 fignotes_dict, chords_dict,
    #                                                 term_states)
    #             print(res)
    #             feat_exp.append(res)