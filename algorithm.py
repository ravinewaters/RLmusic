__author__ = 'redhat'

from common_methods import *
from generate_features import compute_binary_features_expectation
from features_expectation import compute_policy_features_expectation, generate_random_policy_matrix
from math import sqrt
from scipy import sparse, io
from datetime import datetime
from random import random
from cvxopt import matrix, spmatrix, solvers
import numpy as np


def compute_policies(disc_rate, eps):
    value_iteration_n_iter = 30
    value_iteration_error_threshold = 1e-1
    max_reward = 1000
    print('\ndisc_rate', disc_rate)
    print('eps:', eps)
    print('number of iteration of value iteration algorithm:',
          value_iteration_n_iter)
    print('value_iteration_error_threshold:', value_iteration_error_threshold)
    print('terminal states reward:', max_reward)
    all_actions = load_obj('ALL_ACTIONS')
    start_states = load_obj('START_STATES')
    state_size = load_obj('STATE_ELEM_SIZE')

    try:
        q_states = load_obj('Q_STATES')
        # print('Loaded Q_STATES')
    except FileNotFoundError:
        all_states = load_obj('ALL_STATES')
        list_of_all_states = [k+v for k, v in all_states.items()]
        list_of_all_actions = [k+v for k, v in all_actions.items()]
        q_states = generate_all_possible_q_states(list_of_all_states,
                                                  list_of_all_actions)

    try:
        # print('LOAD SAVED STATE.')
        temp = io.loadmat(DIR + 'TEMP')
        policies = temp['policies'].tolist()[0]
        mu_expert = temp['mu_expert']
        mu = temp['mu'].tolist()[0]
        mu_bar = temp['mu_bar']
        counter = temp['counter'][0][0]
    except FileNotFoundError:
        policy_matrix = generate_random_policy_matrix(q_states,
                                                      state_size)
        policies = [policy_matrix]
        mu_expert = compute_expert_features_expectation(disc_rate)
        mu = []
        counter = 1


    print('\n', 'counter, t')
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
        print('{}, {}'.format(counter, t))
        if t <= eps:
            break
        w /= t

        policy_matrix = compute_optimal_policy(w, disc_rate,
                                               value_iteration_error_threshold,
                                               max_reward,
                                               all_actions,
                                               q_states, value_iteration_n_iter)
        policies.append(policy_matrix)
        mu_value = compute_policy_features_expectation(policy_matrix,
                                                       disc_rate,
                                                       start_states,
                                                       1)
        # print('mu_value:', mu_value)
        mu.append(mu_value)
        counter += 1
        io.savemat(DIR + 'TEMP', {'policies': policies,
                            'mu_expert': mu_expert,
                            'mu': mu,
                            'mu_bar': mu_bar,
                            'counter': counter})
        # save policies, mu_expert, mu, mu_bar counter
    mu = [mu_expert] + mu

    # save policies and mu
    save_obj(policies, 'POLICIES')
    save_obj(mu, 'MU')

    # delete TEMP.mat
    if os.path.exists(DIR + 'TEMP.mat'):
        os.remove(DIR + 'TEMP.mat')
    return policies, mu


def compute_projection(mu_bar, mu, mu_expert):
    mu_mu_bar_distance = mu - mu_bar
    mu_bar_mu_expert_distance = mu_expert - mu_bar
    numerator = mu_mu_bar_distance.dot(mu_bar_mu_expert_distance.T)[0, 0]
    denominator = mu_mu_bar_distance.dot(mu_mu_bar_distance.T)[0, 0]
    return numerator/denominator * mu_mu_bar_distance


def compute_optimal_policy(w, disc_rate, eps, max_reward, all_actions,
                           q_states, value_iteration_n_iter):
    term_states = load_obj('TERM_STATES')
    start_states = load_obj('START_STATES')
    state_size = load_obj('STATE_ELEM_SIZE')
    action_size = state_size[:2]
    min_elem, max_elem = load_obj('ELEM_RANGE')
    fignotes_dict = load_obj('FIGNOTES_DICT')
    chords_dict = load_obj('CHORDS_DICT')
    all_actions = load_obj('ALL_ACTIONS')

    # try:
    #     # load saved state
    #     print('Loading saved state.')
    #     temp = io.loadmat(DIR + 'temp')
    #     q_matrix = temp['q_matrix'].todok()
    #     errors = temp['errors'].todok()
    # except Exception as e:
        # print(e)
    q_matrix = dict()
    errors = dict()

    start_states = list(start_states)

    threshold = 2*eps*(1-disc_rate)/disc_rate
    # delta = threshold
    # print('threshold:', threshold)
    iteration = 1
    # while delta >= threshold:
    # number_of_states = 0
    # number_of_states_greater_than_delta = 0
    for i in range(value_iteration_n_iter):
        delta = threshold
        # print('delta:', delta)
        # print('\niteration:', iteration)
        # print('number of states went through:', number_of_states)
        # print('number of states greater than delta:',
        #       number_of_states_greater_than_delta)
        # number_of_states = 0
        # number_of_states_greater_than_delta = 0
        for start_state in start_states:
            trajectory = generate_trajectory_based_on_errors(start_state,
                                                    term_states,
                                                    all_actions,
                                                    q_states,
                                                    errors,
                                                    state_size,
                                                    action_size, .75)
            for j in range(0, len(trajectory), 2):
                # number_of_states += 1
                int_s, state = trajectory[j]
                try:
                    int_a, action = trajectory[j+1]
                except IndexError:
                    # terminal state
                    q_matrix[int_s] = {1: max_reward}
                    break
                # print('\nerrors({}, {}): {}'.format(int_s, int_a,
                #                                     errors[int_s, int_a]))
                feat_exp = compute_binary_features_expectation(state, action,
                                                min_elem, max_elem,
                                                fignotes_dict, chords_dict,
                                                term_states)

                # print('row.size:', row.size)
                if int_s in q_matrix:
                    max_q_value = max(q_matrix[int_s].values())
                    new_q_value = w.dot(feat_exp.T)[0, 0] + disc_rate * max_q_value
                    if int_a not in q_matrix[int_s]:
                        diff = abs(new_q_value)
                    else:
                        diff = abs(new_q_value - q_matrix[int_s][int_a])
                    q_matrix[int_s][int_a] = new_q_value
                    errors[int_s][int_a] = diff
                else:
                    new_q_value = w.dot(feat_exp.T)[0, 0]
                    diff = abs(new_q_value)
                    q_matrix[int_s] = {int_a: new_q_value}
                    errors[int_s] = {int_a: diff}

                if diff > delta:
                    delta = diff
                    # number_of_states_greater_than_delta += 1
                    # print('delta:', delta)

        iteration += 1

    n_rows = np.array(state_size).prod()
    n_cols = np.array(action_size).prod()
    shape = (n_rows, n_cols)

    rows = []
    cols = []
    for int_s in q_matrix:
        rows.append(int_s)
        max_index = dict_argmax(q_matrix[int_s])
        cols.append(max_index)
    data = [1] * len(rows)
    policy_matrix = sparse.coo_matrix((data, (rows, cols)), shape=shape,
                                      dtype=np.uint8)
    return policy_matrix.tocsr()


def dict_argmax(dict):
    return max(dict.items(), key=lambda x: x[1])[0]


def compute_expert_features_expectation(disc_rate):
    term_states = load_obj('TERM_STATES')
    min_elem, max_elem = load_obj('ELEM_RANGE')
    fignotes_dict = load_obj('FIGNOTES_DICT')
    chords_dict = load_obj('CHORDS_DICT')

    trajectories = load_obj('TRAJECTORIES')
    expert_feat_exp = 0
    for trajectory in trajectories:
        sum_of_feat_exp = 0
        t = 0
        # from the first state to the penultimate state
        for i in range(0, len(trajectory)-2, 2):
            state = trajectory[i]
            action = trajectory[i+1]
            feat_exp = compute_binary_features_expectation(state, action,
                                                min_elem, max_elem,
                                                fignotes_dict, chords_dict,
                                                term_states)
            discounted_feat_exp = disc_rate ** t * feat_exp
            sum_of_feat_exp += discounted_feat_exp
            t += 1
        expert_feat_exp += sum_of_feat_exp
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

        if int_s in errors:  # if the row has nonzero entries.
            row = errors[int_s]
            # with prob. gamma, choose random action
            if random() < gamma:
                int_a, action = choose_random_action(q_states, state,
                                                     action_size)
            # with prob. 1-gamma, based on errors. Smaller error,
            # lesser chance to be chosen.
            else:
                # simulate probability
                int_a = weighted_choice(row.items())
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


def choose_policy(policies, mu):
    solvers.options['show_progress'] = False
    n = len(mu) - 1
    A_data = [1]*(n+1)
    A_rows = [0] + [1]*n
    A_cols = range(n+1)
    A = spmatrix(A_data, A_rows, A_cols)
    b = matrix([1.0, 1.0])
    G = spmatrix([-1]*n, range(0, n), range(1, n+1), (n, n+1))
    h = matrix([0.0]*n)
    q = matrix([0.0]*(n+1))
    B = sparse.vstack(mu)
    P = (B * B.T).tocoo()
    P = spmatrix(P.data, P.row, P.col)
    lambdas = list(solvers.qp(2*P, q, G, h, A, b)['x'])[1:]
    print('\n', 'lambdas')
    [print(item) for item in lambdas]
    save_obj(lambdas, 'LAMBDAS')
    return mix_policies(policies, lambdas)


def mix_policies(policies, lambdas):
    opt_pol_index = weighted_choice(zip(range(len(policies)), lambdas))
    save_obj(opt_pol_index, 'OPTIMAL POLICY INDEX')
    return policies[opt_pol_index]


if __name__ == '__main__':
    print('\n')
    print(datetime.now())

    try:
        policies, mu = compute_policies(0.7, 0.7)
        print('\n', 'policies_nnz')
        [print(policy.getnnz()) for policy in policies]
        policy = choose_policy(policies, mu)
        print('\n', datetime.now())

    except KeyboardInterrupt:
        # if os.path.exists(DIR + 'TEMP.mat'):
        #     os.remove(DIR + 'TEMP.mat')
        print('\n', datetime.now())
