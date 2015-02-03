__author__ = 'redhat'

import numpy as np
import random
from scipy import sparse, io
from common_methods import *
from pprint import pprint
from generate_features import compute_binary_features_expectation


def generate_random_policy_matrix(all_states, all_actions, state_elem_size):
    # generate matrix of 0-1 value with size:
    # rows = # states
    # cols = # actions
    reduced_state_size = state_elem_size[:2] + (16,)
    reduced_action_size = state_elem_size[:2]
    n_rows = np.array(reduced_state_size).prod()
    n_cols = np.array(reduced_action_size).prod()
    shape = (n_rows, n_cols)
    policy_matrix = sparse.dok_matrix(shape, dtype=np.uint8)
    for state in all_states:
        action = random.choice(all_actions)
        while not is_valid_action(state, action):
            action = random.choice(all_actions)
        reduced_state = state[:2] + (state[3],)
        reduced_action = action[:2]
        int_s = array_to_int(reduced_state[::-1], reduced_state_size[::-1])
        int_a = array_to_int(reduced_action[::-1], reduced_action_size[::-1])
        policy_matrix[int_s, int_a] = 1
    policy_matrix_csr = policy_matrix.tocsr()
    save_obj(policy_matrix_csr, 'POLICY_0')
    return policy_matrix_csr

def computer_feature_expectation(policy_matrix,
                                 disc_rate, start_state,
                                 term_states, n_iter, all_state):
    min_elem, max_elem = load_obj('ELEM_RANGE')
    fignotes_dict = load_obj('FIGNOTES_DICT')
    # what is s and a? tuple of integers
    mean = 0
    n_action = policy_matrix.shape[1]
    for i in range(n_iter):
        cum_value = 0
        s = start_state
        t = 0
        while s not in term_states:
            row_prob = policy_matrix[s].toarray().ravel()
            a = np.random.choice(n_action,
                                 p=row_prob/sum(row_prob))
            cum_value += disc_rate ** t * \
                         compute_binary_features_expectation(s,
                                                             a,
                                                             min_elem,
                                                             max_elem,
                                                             fignotes_dict,
                                                             chords_dict,
                                                             term_states)
            s = compute_next_state(s, a)
            t += 1
        mean += (cum_value - mean)/i

    return mean

if __name__ == '__main__':
    all_states = load_obj('ALL_STATES')
    all_actions = load_obj('ALL_ACTIONS')
    start_states = load_obj('START_STATES')
    term_states = load_obj('TERM_STATES')
    state_elem_size = load_obj('STATE_ELEM_SIZE')
    policy_matrix = generate_random_policy_matrix(all_states,
                                                  all_actions,
                                                  state_elem_size)
    pprint(policy_matrix)
    # start_state = next(iter(start_states))
    # features_matrix = io.loadmat(DIR + 'FEATURES_MATRIX')['features_matrix']
    # avg_feat_exp = computer_feature_expectation(features_matrix,
    #                                             policy_matrix,
    #                                        .96,
    #                              start_state, term_states, 100,
    #                              states_dict, actions_dict)
    #
    # pprint(avg_feat_exp)