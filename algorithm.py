__author__ = 'redhat'

from common_methods import *
from generate_features import compute_binary_features_expectation
from features_expectation import compute_policy_features_expectation, generate_random_policy_matrix
from math import sqrt
from scipy import sparse
import numpy as np
from datetime import datetime

def compute_policies(disc_rate, eps):
    all_states = load_obj('ALL_STATES')
    all_actions = load_obj('ALL_ACTIONS')
    start_states = load_obj('START_STATES')
    state_size = load_obj('STATE_ELEM_SIZE')
    policy_matrix_0 = generate_random_policy_matrix(all_states,
                                                  all_actions,
                                                  state_size)
    mu_expert = compute_expert_features_expectation(disc_rate)
    mu = []
    policies = []
    counter = 1
    while True:
        if counter == 1:
            mu_value = compute_policy_features_expectation(policy_matrix_0,
                                                               disc_rate,
                                                               start_states,
                                                               1,)
            mu.append(mu_value)
            mu_bar = mu[counter-1]  # mu_bar[0] = mu[0]
        else:  # counter >= 2
            mu_bar += compute_projection(mu_bar, mu, mu_expert)

        w = mu_expert - mu_bar
        t = sqrt((w.data**2).sum())
        if t <= eps:
            break
        w /= t  # normalize w
        print('w:', w)
        policy_matrix = q_value_iteration_algorithm(w, 0.99, 1e-2, 1000)
        policies.append(policy_matrix)
        mu_value = compute_policy_features_expectation(policy_matrix,
                                                               disc_rate,
                                                               start_states,
                                                               1,)
        print('mu_value:', mu_value)
        mu.append(mu_value)
        counter += 1
        print('counter:', counter)
    return policies, mu


def compute_projection(mu_bar, mu, mu_expert):
    mu_mu_bar_distance = mu - mu_bar
    mu_bar_mu_expert_distance = mu_expert - mu_bar
    numerator = np.dot(mu_mu_bar_distance, mu_bar_mu_expert_distance)
    denominator = np.dot(mu_mu_bar_distance, mu_mu_bar_distance)
    return numerator/denominator * mu_mu_bar_distance


def q_value_iteration_algorithm(w, disc_rate, eps, max_reward):
    all_states = load_obj('ALL_STATES')
    all_actions = load_obj('ALL_ACTIONS')
    min_elem, max_elem = load_obj('ELEM_RANGE')
    fignotes_dict = load_obj('FIGNOTES_DICT')
    chords_dict = load_obj('CHORDS_DICT')
    term_states = load_obj('TERM_STATES')
    state_size = load_obj('STATE_ELEM_SIZE')

    all_actions = [k+v for k, v in all_actions.items()]
    all_states = [k+v for k, v in all_states.items()]
    action_size = state_size[:2]
    n_rows = np.array(state_size).prod()
    n_cols = np.array(action_size).prod()
    shape = (n_rows, n_cols)
    policy_matrix = sparse.dok_matrix(shape)
    Q2 = sparse.dok_matrix(shape)

    delta = 0
    while delta < eps*(1-disc_rate)/disc_rate:
        Q1 = Q2.copy()
        delta = 0
        for state in all_states:
            for action in all_actions:
                reduced_state = state[:3]
                reduced_action = action[:2]
                int_s = array_to_int(reduced_state[::-1],
                                     state_size[::-1])
                print('int_s:', int_s)
                int_a = array_to_int(reduced_action[::-1],
                                     action_size[::-1])
                print('int_a:', int_a)
                if state in term_states:
                    Q2[int_s, 1] = max_reward
                    print('max_reward:', max_reward)
                else:
                    feat_exp = compute_binary_features_expectation(state,
                                                                   action,
                                                                   min_elem,
                                                                   max_elem,
                                                                   fignotes_dict,
                                                                   chords_dict,
                                                                   term_states)

                    row = Q1[int_s].tocsr()
                    if row.size:
                        max_q_value = max(row.data)
                    else:
                        max_q_value = 0
                    Q2[int_s, int_a] = w.dot(feat_exp.T)[0, 0] + \
                                       disc_rate * max_q_value
                    print('Q2[int_s, int_a]:', Q2[int_s, int_a])
                    diff = Q2[int_s, int_a] - Q1[int_s, int_a]
                    print('diff:', diff)
                    if abs(diff) > delta:
                        delta = diff
                        print('delta:', delta)

    Q2 = Q2.tocsc()
    for int_s in np.unique(Q2.indices):
        row = Q2[int_s]
        max_index = row.indices[row.data.argmax()] if row.nnz else 0
        policy_matrix[int_s, max_index] = 1

    return policy_matrix


def compute_expert_features_expectation(disc_rate):
    min_elem, max_elem = load_obj('ELEM_RANGE')
    fignotes_dict = load_obj('FIGNOTES_DICT')
    chords_dict = load_obj('CHORDS_DICT')
    term_states = load_obj('TERM_STATES')
    trajectories = load_obj('TRAJECTORIES')
    expert_feat_exp = 0
    for trajectory in trajectories:
        feat_exp = 0
        t = 0
        # from the first state to the penultimate state
        for i in range(0, len(trajectory)-2, 2):
            state = trajectory[i]
            action = trajectory[i+1]
            feat_exp += disc_rate ** t * compute_binary_features_expectation(
                state,
                action,
                min_elem,
                max_elem,
                fignotes_dict,
                chords_dict,
                term_states)
            t += 1
        expert_feat_exp += feat_exp
    expert_feat_exp /= len(trajectories)
    return expert_feat_exp

if __name__ == '__main__':
    # print(compute_expert_features_expectation(0.99))
    print(datetime.now())

    compute_policies(0.99, 1e-3)

    print(datetime.now())