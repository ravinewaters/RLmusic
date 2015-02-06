__author__ = 'redhat'

from common_methods import *
from generate_features import compute_binary_features_expectation
from features_expectation import compute_policy_features_expectation, generate_random_policy_matrix
from math import sqrt
from scipy import sparse, io
from datetime import datetime
# from itertools import product
from random import shuffle, choice, uniform
from pprint import pprint
import numpy as np

def compute_policies(disc_rate, eps):
    all_actions = load_obj('ALL_ACTIONS')
    start_states = load_obj('START_STATES')
    state_size = load_obj('STATE_ELEM_SIZE')

    try:
        q_states = load_obj('Q_STATES')
        print('Loaded Q_STATES')
    except IOError:
        all_states = load_obj('ALL_STATES')
        list_of_all_states = [k+v for k, v in all_states.items()]
        list_of_all_actions = [k+v for k, v in all_actions.items()]
        q_states = generate_all_possible_q_states(list_of_all_states,
                                                  list_of_all_actions)

    policy_matrix = generate_random_policy_matrix(q_states,
                                                  state_size)
    policies = [policy_matrix]
    mu_expert = compute_expert_features_expectation(disc_rate)
    mu = []
    counter = 1
    while True:
    # for i in range(2):
        if counter == 1:
            mu_value = compute_policy_features_expectation(policy_matrix,
                                                               disc_rate,
                                                               start_states,
                                                               1,)
            mu.append(mu_value)
            mu_bar = mu[counter-1]  # mu_bar[0] = mu[0]
        else:  # counter >= 2
            mu_bar += compute_projection(mu_bar, mu[counter-1], mu_expert)

        w = mu_expert - mu_bar
        t = sqrt((w.data**2).sum())
        if t <= eps:
            break
        w /= t  # normalize w
        # print('w:', w)
        policy_matrix = compute_optimal_policy(w, disc_rate,
                                               1e-1, 1000,
                                               all_actions,
                                               q_states)
        policies.append(policy_matrix)
        mu_value = compute_policy_features_expectation(policy_matrix,
                                                               disc_rate,
                                                               start_states,
                                                               )
        # print('mu_value:', mu_value)
        mu.append(mu_value)
        counter += 1
        print('counter:', counter)
        # save policies, mu_expert, mu, mu_bar counter

    return policies, mu


def compute_projection(mu_bar, mu, mu_expert):
    mu_mu_bar_distance = mu - mu_bar
    mu_bar_mu_expert_distance = mu_expert - mu_bar
    numerator = mu_mu_bar_distance.dot(mu_bar_mu_expert_distance.T)[0, 0]
    denominator = mu_mu_bar_distance.dot(mu_mu_bar_distance.T)[0, 0]
    return numerator/denominator * mu_mu_bar_distance


def compute_optimal_policy(w, disc_rate, eps, max_reward, all_actions,
                           q_states):
    min_elem, max_elem = load_obj('ELEM_RANGE')
    fignotes_dict = load_obj('FIGNOTES_DICT')
    chords_dict = load_obj('CHORDS_DICT')
    term_states = load_obj('TERM_STATES')
    start_states = load_obj('START_STATES')
    state_size = load_obj('STATE_ELEM_SIZE')
    action_size = state_size[:2]

    n_rows = np.array(state_size).prod()
    n_cols = np.array(action_size).prod()
    shape = (n_rows, n_cols)
    try:
        # load saved state
        print('Loading saved state.')
        temp = io.loadmat(DIR + 'temp')
        q_matrix = temp['q_matrix'].todok()
        errors = temp['errors'].todok()
    except Exception as e:
        # print(e)
        q_matrix = sparse.dok_matrix(shape)
        errors = sparse.dok_matrix(shape, dtype=np.float64)

    start_states = list(start_states)
    shuffle(start_states)

    threshold = 2*eps*(1-disc_rate)/disc_rate
    delta = threshold
    # print('threshold:', threshold)
    iteration = 0
    while delta >= threshold:
    # for i in range(1):
        delta = threshold
        print('iteration:', iteration)
        print('delta:', delta)
        for start_state in start_states:
            trajectory = generate_trajectory_based_on_errors(start_state,
                                                    term_states,
                                                    all_actions,
                                                    q_states,
                                                    errors,
                                                    state_size,
                                                    action_size, .05)
            for i in range(0, len(trajectory), 2):
                int_s, state = trajectory[i]
                try:
                    int_a, action = trajectory[i+1]
                except IndexError:
                    # terminal state
                    q_matrix[int_s, 1] = max_reward
                    # print('terminal state:', int_s)
                    # print('max_reward:', max_reward)
                    break
                # print('\nerrors({}, {}): {}'.format(int_s, int_a,
                #                                     errors[int_s, int_a]))
                feat_exp = compute_binary_features_expectation(state,
                                                               action,
                                                               min_elem,
                                                               max_elem,
                                                               fignotes_dict,
                                                               chords_dict,
                                                               term_states)

                row = q_matrix[int_s].tocsr()
                # print('row.size:', row.size)
                if row.size != 0:
                    max_q_value = max(row.data)
                    # print('max_q_value:', max_q_value)
                else:
                    max_q_value = 0
                new_q_value = w.dot(feat_exp.T)[0, 0] + disc_rate * max_q_value
                # print('old q_value:', q_matrix[int_s, int_a])
                # print('new q_value:', new_q_value)
                diff = abs(new_q_value - q_matrix[int_s, int_a])
                errors[int_s, int_a] = diff
                # print('diff:', diff)
                q_matrix[int_s, int_a] = new_q_value

                if diff > delta:
                    delta = diff
                    print('delta:', delta)

        # save state
        io.savemat(DIR + 'temp', {'q_matrix': q_matrix, 'errors': errors})
        iteration += 1

    q_matrix = q_matrix.tocoo()
    policy_matrix = sparse.dok_matrix(shape)
    for int_s in np.unique(q_matrix.row):
        row = q_matrix.getrow(int_s)
        max_index = row.indices[row.data.argmax()] if row.nnz else 0
        # print('(int_s, max_index) = ({}, {})'.format(int_s, max_index))
        policy_matrix[int_s, max_index] = 1

    return policy_matrix.tocsr()


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

def generate_trajectory_based_on_errors(state, term_states,
                               all_actions, q_states, errors, state_size,
                               action_size, gamma):
    # original state
    # all_actions dict
    trajectory = []
    while state not in term_states:
        int_s = array_to_int(state[:3][::-1], state_size[::-1])
        trajectory.append((int_s, state))

        row_csr = errors[int_s].tocsr()
        if row_csr.size:  # if the row has nonzero entries.
            
            # with prob. gamma, choose random action
            if uniform(0, 1) < gamma:
                int_a, action = choose_random_action(q_states, state,
                                                     action_size)
            # with prob. 1-gamma, based on errors. Smaller error,
            # lesser chance to be chosen.
            else:
                indices = row_csr.indices
                row = row_csr.data * 10
                probs = row/sum(row)
                int_a = random_pick(indices, probs)
                key_a = tuple(int_to_array(int_a, action_size[::-1])[::-1])
                action = key_a + all_actions[key_a]
        else:
            int_a, action = choose_random_action(q_states, state,
                                                 action_size)
        trajectory.append((int_a, action))
        state = compute_next_state(state, action)

    int_s = array_to_int(state[:3][::-1], state_size[::-1])
    trajectory.append((int_s, state))  # append terminal state
    return trajectory


def random_pick(choices, probs):
    cutoffs = np.cumsum(probs)
    idx = cutoffs.searchsorted(np.random.uniform(0, cutoffs[-1]))
    return choices[idx]


def generate_all_possible_q_states(all_states, all_actions):
    # assume complete states and actions, not reduced ones.
    q_states = {}
    for state in all_states:
        for action in all_actions:
            if is_valid_action(state, action):
                if state in q_states:
                    q_states[state].append(action)
                else:
                    q_states[state] = [action]
    for state in q_states:
        q_states[state] = tuple(q_states[state])
    save_obj(q_states, 'Q_STATES')
    return q_states




def choose_policy(policies, mu):
    pass

if __name__ == '__main__':
    # print(compute_expert_features_expectation(0.99))
    print(datetime.now())

    policies, mu = compute_policies(0.7, 1e-1)
    pprint(policies)
    pprint(mu)

    print(datetime.now())
