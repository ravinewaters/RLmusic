__author__ = 'redhat'

from scipy import sparse, io
from common_methods import *
from generate_features import compute_binary_features_expectation


def generate_random_policy_matrix(q_states):
    # generate matrix of 0-1 value with size:
    # rows = # states
    # cols = # actions
    # Use dictionary not matrix.
    # not stochastic
    # Should add stochastic policy to the matrix.

    policy_matrix = {k: ((choice(v), 1.0),) for k, v in q_states.items()}
    return policy_matrix


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

    min_elem, max_elem = load_obj('ELEM_RANGE')
    fignotes_dict = load_obj('FIGNOTES_DICT')
    chords_dict = load_obj('CHORDS_DICT')
    term_states = load_obj('TERM_STATES')
    state = choice(list(start_states))
    # what is s and a? tuple of integers

    mean_feat_exp = 0
    for i in range(n_iter):
        sum_of_feat_exp = 0
        t = 0
        while state not in term_states:
            action = weighted_choice(policy_matrix[state])
            feat_exp = compute_binary_features_expectation(state, action,
                                                min_elem, max_elem,
                                                fignotes_dict, chords_dict,
                                                term_states)
            discounted_feat_exp = disc_rate ** t * feat_exp
            sum_of_feat_exp += discounted_feat_exp
            if discounted_feat_exp.sum() <= 1e-5:
                break
            state = compute_next_state(state, action)
            t += 1
        mean_feat_exp += sum_of_feat_exp
    return mean_feat_exp/n_iter


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