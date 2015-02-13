__author__ = 'redhat'

from common_methods import *
from generate_features import compute_binary_features_expectation
from random import choice

def generate_random_policy_matrix(q_states):
    # generate matrix of 0-1 value with size:
    # rows = # states
    # cols = # actions
    # Use dictionary not matrix.
    # not stochastic
    # Should add stochastic policy to the matrix.

    policy_matrix = {s: ((choice(list(v)), 1),) for s, v in q_states.items()}
    return policy_matrix


def compute_policy_features_expectation(feat_mtx, q_states, policy_matrix,
                                        disc_rate, start_states):
    # Basically what the function does is walk through the states and
    # actions. The actions are gotten by choosing randomly according to the
    # policy matrix. We start from a given start_state and stop when
    # reaching a terminal state or the features expectation is very small
    # because of the discount factor.
    # We generate n_iter trajectories and find the average of the sum of
    # discounted feature expectation. If we have a deterministic policy we
    # can just set the n_iter to be 1.

    # policy_matrix:
    # {state: ((a1, .05), (a2, .1), (a3, .85))}


    term_states = load_obj('TERM_STATES')

    counter = 1
    mean_feat_exp = 0
    for state in start_states:
        sum_of_feat_exp = 0
        t = 0
        while True:
            print('state:', state)
            action = weighted_choice(policy_matrix[state])
            row = q_states[state][action][0]
            feat_exp = feat_mtx[row]
            discounted_feat_exp = disc_rate ** t * feat_exp
            sum_of_feat_exp += discounted_feat_exp

            if state in term_states and action == -1:
                break
            elif discounted_feat_exp.sum() <= 1e-5:
                break
            state = compute_next_state(state, action)
            t += 1
        mean_feat_exp += sum_of_feat_exp
        counter += 1

    return mean_feat_exp/counter


if __name__ == '__main__':
    pass