__author__ = 'redhat'

import numpy as np
import random
from scipy import sparse, io
from common_methods import *
from generate_features import generate_binary_features_expectation_table


def generate_random_policy_matrix(q_states, state_size):
    # generate matrix of 0-1 value with size:
    # rows = # states
    # cols = # actions
    # Should add stochastic policy to the matrix.

    rows = []
    cols = []
    for state in q_states:
        int_a, _ = choose_random_action(q_states, state, state_size[:2])
        int_s = array_to_int(state[:3][::-1], state_size[::-1])
        rows.append(int_s)
        cols.append(int_a)
    data = [1]*len(cols)

    n_rows = np.array(state_size).prod()
    n_cols = np.array(state_size[:2]).prod()
    shape = (n_rows, n_cols)
    policy_matrix = sparse.coo_matrix((data, (rows, cols)), shape=shape,
                                      dtype=np.uint8)

    return policy_matrix.tocsr()


def compute_policy_features_expectation(policy_matrix, disc_rate, start_states,
                                        n_iter=1):
    # Basically what the function does is walk through the states and
    # actions. The actions are gotten by choosing randomly according to the
    # policy matrix. We start from a given start_state and stop when
    # reaching a terminal state or the features expectation is very small
    # because of the discount factor.
    # We generate n_iter trajectories and find the average of the sum of
    # discounted feature expectation. If we have a deterministic policy we
    # can just set the n_iter to be 1.

    try:
        features_expectation_dict = load_obj('FEATURES_EXPECTATION_DICT')
    except FileNotFoundError:
        features_expectation_dict = generate_binary_features_expectation_table()

    term_states = load_obj('TERM_STATES')
    state_size = load_obj('STATE_ELEM_SIZE')
    all_actions = load_obj('ALL_ACTIONS')
    state = random.choice(list(start_states))
    action_size = state_size[:2]
    # what is s and a? tuple of integers
    mean_feat_exp = 0
    for i in range(n_iter):
        # print('i=', i)
        sum_of_feat_exp = 0
        t = 0
        while state not in term_states:
            # print('state:', state)

            action = choose_action_from_state(policy_matrix, all_actions,
                                              state, state_size, action_size)
            # print('action', action)
            feat_exp = disc_rate ** t * \
                       np.array(features_expectation_dict[(state, action)])
            sum_of_feat_exp += feat_exp
            if feat_exp.sum() <= 1e-5:
                # print('break:', t)
                break
            # print('feat_exp:', feat_exp)
            state = compute_next_state(state, action)
            t += 1
        # print(t)
        mean_feat_exp += sum_of_feat_exp
    # print('mean:', mean_feat_exp)
    return mean_feat_exp/n_iter

def choose_action_from_state(policy_matrix, all_actions,
                          state, state_size, action_size):
    # doesn't check whether state is a terminal state
    reduced_state = state[:3]
    int_s = array_to_int(reduced_state[::-1], state_size[::-1])
    indices = policy_matrix[int_s].indices
    row = policy_matrix[int_s].data
    prob = row/sum(row)
    int_a = np.random.choice(indices, p=prob)  # int
    key_a = tuple(int_to_array(int_a, action_size[::-1])[::-1])
    # print(key_a)
    action = key_a + all_actions[key_a]
    return action

if __name__ == '__main__':
    all_states = load_obj('ALL_STATES')
    all_actions = load_obj('ALL_ACTIONS')
    start_states = load_obj('START_STATES')

    policy_matrix = generate_random_policy_matrix(all_states,
                                                  all_actions,
                                                  state_elem_size)
    avg_feat_exp = compute_policy_features_expectation(policy_matrix,
                                               0.99,
                                               start_states,
                                               1,)

    print(avg_feat_exp)