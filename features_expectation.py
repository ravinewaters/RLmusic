__author__ = 'redhat'

import numpy as np
from scipy import sparse, io
from common_methods import *
from pprint import pprint

def generate_possible_action(state_action_dict):
    possible_action = {}
    for state_action in state_action_dict[0]:
        if state_action[0] in possible_action:
            possible_action[state_action[0]].append(state_action[1])
        else:
            possible_action[state_action[0]] = [state_action[1]]
    return possible_action

def generate_random_policy_matrix(possible_action_dict):
    n_rows = len(states_dict[0])
    n_cols = len(actions_dict[0])
    shape = (n_rows, n_cols)
    policy_matrix = sparse.dok_matrix(shape)
    for int_s in possible_action_dict:
        int_a = np.random.choice(possible_action_dict[int_s])
        policy_matrix[int_s, int_a] = 1
    return policy_matrix.tocsr()

def computer_feature_expectation(features_matrix, policy_matrix,
                                 disc_rate, start_state,
                                 terminal_states, n_iter,
                                 states_dict,
                                 actions_dict):
    # what is s and a? int
    mean = 0
    n_action = policy_matrix.shape[1]
    for i in range(n_iter):
        cum_value = 0
        s = start_state
        t = 0
        while s not in terminal_states:
            row_prob = policy_matrix[s].toarray().ravel()
            a = np.random.choice(n_action,
                                 p=row_prob/sum(row_prob))
            cum_value += disc_rate ** t * features_matrix[array_to_int((s, a),
                                                                       policy_matrix.shape)]
            s_prime = compute_next_state(states_dict[1][s], actions_dict[1][a])
            print(s_prime)
            s = states_dict[0][s_prime]
            t += 1
        mean += + (cum_value - mean)/i

    return mean

if __name__ == '__main__':
    states_dict = load_obj('STATES_DICT')
    actions_dict = load_obj('ACTIONS_DICT')
    term_states = load_obj('TERM_STATES')
    start_states = load_obj('START_STATES')
    state_action_dict = load_obj('STATE_ACTION_DICT')
    possible_action_dict = generate_possible_action(state_action_dict)
    save_obj(possible_action_dict, 'POSSIBLE_ACTION_DICT')
    policy_matrix = generate_random_policy_matrix(possible_action_dict)

    start_state = next(iter(start_states))
    features_matrix = io.loadmat(DIR + 'FEATURES_MATRIX')['features_matrix']
    avg_feat_exp = computer_feature_expectation(features_matrix,
                                                policy_matrix,
                                           .96,
                                 start_state, term_states, 100,
                                 states_dict, actions_dict)

    pprint(policy_matrix)
    pprint(avg_feat_exp)