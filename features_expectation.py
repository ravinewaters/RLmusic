__author__ = 'redhat'

import numpy as np
import random
from scipy import sparse, io
from common_methods import *
from pprint import pprint
from generate_features import compute_binary_features_expectation


def generate_random_policy_matrix(all_states, all_actions, state_size):
    # generate matrix of 0-1 value with size:
    # rows = # states
    # cols = # actions
    all_actions = [k+v for k, v in all_actions.items()]
    action_size = state_size[:2]

    n_rows = np.array(state_size).prod()
    n_cols = np.array(action_size).prod()
    shape = (n_rows, n_cols)

    policy_matrix = sparse.dok_matrix(shape, dtype=np.uint8)
    for first_three, last_two in all_states.items():
        # print('state:', first_three+last_two)
        action = random.choice(all_actions)
        while not is_valid_action(first_three + last_two, action):
            action = random.choice(all_actions)
        reduced_action = action[:2]
        # print('action:', action)
        int_s = array_to_int(first_three[::-1], state_size[::-1])
        int_a = array_to_int(reduced_action[::-1], action_size[::-1])
        policy_matrix[int_s, int_a] = 1
    policy_matrix_csr = policy_matrix.tocsr()
    save_obj(policy_matrix_csr, 'POLICY_0')
    return policy_matrix_csr


def computer_feature_expectation(policy_matrix, disc_rate, start_state,
                                 term_states, n_iter):
    # SLOW
    min_elem, max_elem = load_obj('ELEM_RANGE')
    fignotes_dict = load_obj('FIGNOTES_DICT')
    chords_dict = load_obj('CHORDS_DICT')
    state_size = load_obj('STATE_ELEM_SIZE')
    all_actions = load_obj('ALL_ACTIONS')
    action_size = state_size[:2]
    # what is s and a? tuple of integers
    mean = []
    for i in range(n_iter):
        print('i=', i)
        sum_of_feat_exp = 0
        state = start_state
        t = 0
        while state not in term_states:
            # print('state:', state)
            reduced_state = state[:3]
            int_s = array_to_int(reduced_state[::-1], state_size[::-1])
            indices = policy_matrix[int_s].indices
            row = policy_matrix[int_s].data
            prob = row/sum(row)
            a = np.random.choice(indices, p = prob)  # int
            key_a = tuple(int_to_array(a, action_size[::-1])[::-1])
            action = key_a + all_actions[key_a]
            # print('action', action)
            feat_exp = disc_rate ** t * \
                         compute_binary_features_expectation(state,
                                                             action,
                                                             min_elem,
                                                             max_elem,
                                                             fignotes_dict,
                                                             chords_dict,
                                                             term_states)
            sum_of_feat_exp += feat_exp
            if np.all(feat_exp < 1e-7):
                print(t)
                break
            # print('feat_exp:', feat_exp)
            state = compute_next_state(state, action)
            t += 1
        mean.append(sum_of_feat_exp)
    # print('mean:', mean)
    return np.mean(mean, axis=0)

if __name__ == '__main__':
    all_states = load_obj('ALL_STATES')
    all_actions = load_obj('ALL_ACTIONS')
    start_states = load_obj('START_STATES')
    term_states = load_obj('TERM_STATES')
    state_elem_size = load_obj('STATE_ELEM_SIZE')
    policy_matrix = generate_random_policy_matrix(all_states,
                                                  all_actions,
                                                  state_elem_size)
    start_state = next(iter(start_states))
    # features_matrix = io.loadmat(DIR + 'FEATURES_MATRIX')['features_matrix']
    avg_feat_exp = computer_feature_expectation(policy_matrix,
                                                0.99,
                                                start_state,
                                                term_states, 1,
                                                )

    pprint(avg_feat_exp)