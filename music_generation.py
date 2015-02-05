__author__ = 'redhat'
import numpy as np
from common_methods import *
from features_expectation import choose_action_from_state
import music21


def translate_states_to_song(states):
    # use music21
    pass



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


def generate_sequence_of_states(state, term_states,
                               all_actions, policies, lambdas):
    state_size = load_obj('STATE_ELEM_SIZE')
    action_size = state_size[:2]
    policy_matrix = mix_policy(policies, lambdas)
    states = []
    while state not in term_states:
        states.append(state)
        action = choose_action_from_state(policy_matrix, all_actions, state,
                                          state_size, action_size)
        state = compute_next_state(state, action)
    states.append(state)  # append terminal state
    save_obj(states, 'GENERATED_SEQUENCE_OF_STATES')
    return states

def mix_policy(policies, lambdas):
    # input: list of policies and its weight
    # idea: randomly select policy according to its weights (lambdas)
    # output: a policy matrix
    index = np.random.choice(len(policies), p=lambdas)
    return policies[index]