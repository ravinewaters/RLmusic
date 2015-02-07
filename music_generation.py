__author__ = 'redhat'

from common_methods import *
from features_expectation import choose_action_from_policy_matrix
from music21 import note, stream


def generate_trajectory(start_state, term_states, all_actions,
                                policy_matrix):
    state = start_state
    state_size = load_obj('STATE_ELEM_SIZE')
    action_size = state_size[:2]
    states = []
    while state not in term_states:
        states.append(state)
        action = choose_action_from_policy_matrix(policy_matrix,
                                                  all_actions,
                                                  state,
                                                  state_size,
                                                  action_size)
        state = compute_next_state(state, action)
    states.append(state)  # append terminal state
    save_obj(states, 'GENERATED_SEQUENCE_OF_STATES')
    return states


def get_original_state(states, fignotes_dict, chords_dict):
    """
    returns states that are tuple of nonnegative integers, e.g.
    (1,1,1,1,2) instead of ((0, 1.0), 'C', 1.0, 1.0, 0)
    """

    new_states = []
    for state in states:
        new_state = [fignotes_dict[1][state[0]],
                     chords_dict[1][state[1]],
                     int(state[2] / 4.0),
                     int(state[3] / 4.0),
                     state[4]]
        new_states.append(tuple(new_state))
    return new_states





def translate_states_to_song(original_states):
    # use music21

    for state in original_states:
        pitches = state[0][::2]
        durations = state[0][1::2]
        for item in zip(pitches, durations):
            n = note.Note()
            n.pitch


    pass