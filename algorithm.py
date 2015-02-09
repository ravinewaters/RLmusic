__author__ = 'redhat'

from common_methods import *
from generate_features import generate_features_expectation_table
from features_expectation import compute_policy_features_expectation, \
    generate_random_policy_matrix
from math import sqrt
from scipy import sparse, io
# from datetime import datetime
# from time import time
from cvxopt import matrix, spmatrix, solvers

# consider of using dictionary as policy_matrix

def compute_policies(disc_rate, eps):
    max_reward = 1000
    print('\ndisc_rate', disc_rate)
    print('eps:', eps)
    print('terminal states reward:', max_reward)
    all_actions = load_obj('ALL_ACTIONS')
    start_states = load_obj('START_STATES')

    try:
        # required files
        feat_mtx = io.loadmat(DIR + 'FEATURES_EXPECTATION_MATRIX')['mtx']
        q_states = load_obj('Q_STATES')
    except FileNotFoundError:
        feat_mtx = generate_features_expectation_table()
        all_states = load_obj('ALL_STATES')
        list_of_all_states = [k+v for k, v in all_states.items()]
        list_of_all_actions = [k+v for k, v in all_actions.items()]
        q_states = generate_all_possible_q_states(list_of_all_states,
                                                  list_of_all_actions)
                                                  
    try:
        # Load computation
        temp = io.loadmat(DIR + 'TEMP')
        policies = load_obj('TEMP_POLICIES')
        mu_expert = temp['mu_expert']
        mu = temp['mu'].tolist()[0]
        mu_bar = temp['mu_bar']
        counter = temp['counter'][0][0]
    except FileNotFoundError:
        policy_matrix = generate_random_policy_matrix(q_states)
        policies = [policy_matrix]
        mu_expert = compute_expert_features_expectation(feat_mtx, q_states,
                                                        disc_rate)
        mu = []
        counter = 1

    print('\n', 'counter, t')

    for k in range(10):
        if counter == 1:
            mu_value = compute_policy_features_expectation(
                feat_mtx, q_states, policy_matrix,
                                                               disc_rate,
                                                               start_states)
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
        reward_mtx = compute_reward_mtx(feat_mtx, w)
        policy_matrix = compute_optimal_policy(reward_mtx, q_states,
                                               disc_rate, eps,
                                               max_reward)
        policies.append(policy_matrix)
        mu_value = compute_policy_features_expectation(feat_mtx,
                                                       q_states,
                                                       policy_matrix,
                                                       disc_rate,
                                                       start_states)
        mu.append(mu_value)
        counter += 1
        
        # save mu_expert, mu, mu_bar counter, policies for later computation
        io.savemat(DIR + 'TEMP', {'mu_expert': mu_expert,
                            'mu': mu,
                            'mu_bar': mu_bar,
                            'counter': counter})
        save_obj(policies, 'TEMP_POLICIES')

    mu = [mu_expert] + mu

    # save policies and mu
    save_obj(policies, 'POLICIES')
    save_obj(mu, 'MU')

    # delete TEMP.mat
    if os.path.exists(DIR + 'TEMP.mat'):
        os.remove(DIR + 'TEMP.mat')
        os.remove(DIR + 'TEMP_POLICIES.pkl')
    return policies, mu


def compute_projection(mu_bar, mu, mu_expert):
    mu_mu_bar_distance = mu - mu_bar
    mu_bar_mu_expert_distance = mu_expert - mu_bar
    numerator = mu_mu_bar_distance.dot(mu_bar_mu_expert_distance.T)[0, 0]
    denominator = mu_mu_bar_distance.dot(mu_mu_bar_distance.T)[0, 0]
    return numerator/denominator * mu_mu_bar_distance


def compute_reward_mtx(feat_mtx, w):
    return (feat_mtx * w.T).data


def compute_optimal_policy(reward_mtx, q_states, disc_rate, eps, max_reward):
    # start_states = load_obj('START_STATES')
    # start_states = list(start_states)
    q_matrix = {}
    max_values = dict.fromkeys(list(q_states), (0, 0))
    threshold = eps*(1-disc_rate)/disc_rate
    delta = threshold
    print('threshold:', threshold)
    iteration = 1
    while delta >= threshold:
        print('iteration:', iteration)
        delta = -1
        for state, actions in q_states.items():
            for action in actions:

                if action == -1:
                    # if action 'exit'
                    if (state, action) in q_matrix:
                        q_matrix[(state, action)] = max_reward
                    max_values[state] = (max_reward, action)
                    continue

                row = q_states[state][action]
                reward = reward_mtx[row]
                state_prime = compute_next_state(state, action)
                opt_future_val = max_values[state_prime][0]
                new_q_value = reward + disc_rate * opt_future_val
                if (state, action) not in q_matrix:
                    diff = abs(new_q_value)
                    q_matrix[(state, action)] = new_q_value
                else:
                    # if (state, action) in q_matrix
                    diff = abs(new_q_value - q_matrix[(state, action)])
                    q_matrix[(state, action)] = new_q_value

                # update max_values
                if max_values[state][0] < new_q_value:
                    max_values[state] = (new_q_value, action)

                if diff > delta:
                    delta = diff
        iteration += 1
        print('delta', delta)
    policy_matrix = {k: ((v[1], 1.0),) for k, v in max_values.items()}
    return policy_matrix


def dict_argmax(dict):
    return max(dict.items(), key=lambda x: x[1])[0]


def compute_expert_features_expectation(feat_mtx, q_states, disc_rate):
    trajectories = load_obj('TRAJECTORIES')
    expert_feat_exp = 0
    for trajectory in trajectories:
        sum_of_feat_exp = 0
        t = 0
        # from the first state to the penultimate state
        for state, action in zip(trajectory[::2], trajectory[1::2]):
            if action == -1:
                break
            row = q_states[state][action]
            feat_exp = feat_mtx[row]
            discounted_feat_exp = disc_rate ** t * feat_exp
            sum_of_feat_exp += discounted_feat_exp
            t += 1
        expert_feat_exp += sum_of_feat_exp
    expert_feat_exp /= len(trajectories)
    return expert_feat_exp

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
    save_obj(lambdas, 'LAMBDAS')
    return mix_policies(policies, lambdas)


def mix_policies(policies, lambdas):
    opt_pol_index = weighted_choice(zip(range(len(policies)), lambdas))
    save_obj(opt_pol_index, 'OPTIMAL_POLICY_INDEX')
    return policies[opt_pol_index]


if __name__ == '__main__':
    # print('\n')
    # print('\n', datetime.now())

    try:
        # start_time = time()
        policies, mu = compute_policies(0.99, 0.7)
        # end_time = time()
        # duration = end_time - start_time
        # print(duration)

        # print('\n', 'policies_nnz')
        # [print(len(policy)) for policy in policies]

        policy = choose_policy(policies, mu)

    except KeyboardInterrupt:
        # print('\n', datetime.now())
        pass

