__author__ = 'redhat'

from common_methods import *
from generate_features import generate_features_expectation_table
from features_expectation import compute_policy_features_expectation, \
    generate_random_policy_matrix
from math import sqrt, exp
from scipy import sparse, io
from cvxopt import matrix, spmatrix, solvers
from random import choice


def compute_policies(disc_rate, eps):
    # max_reward = 6
    # value_error = 0.01
    print('\ndisc_rate', disc_rate)
    print('eps:', eps)
    # print('value_error:', value_error)
    # print('terminal states reward:', max_reward)

    start_states = load_obj('START_STATES')
    term_states = load_obj('TERM_STATES')

    try:
        # load feat_mtx and q_states
        feat_mtx = io.loadmat(DIR + 'FEATURES_EXPECTATION_MATRIX')['mtx']
        q_states = load_obj('Q_STATES')
    except FileNotFoundError:
        feat_mtx = generate_features_expectation_table()
        all_states = load_obj('ALL_STATES')
        all_actions = load_obj('ALL_ACTIONS')
        q_states = generate_all_possible_q_states(all_states,
                                                  all_actions)
                                                  
    try:
        # Load saved computation state
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
    t = 0
    while counter <= 30:
        if counter == 1:
            mu_value = compute_policy_features_expectation(feat_mtx,
                                                           q_states,
                                                           policy_matrix,
                                                           disc_rate,
                                                           start_states,
                                                           term_states)
            mu.append(mu_value)
            mu_bar = mu[counter-1]  # mu_bar[0] = mu[0]

        else:  # counter >= 2
            mu_bar += compute_projection(mu_bar, mu[counter-1], mu_expert)

        w = mu_expert - mu_bar
        temp = t
        t = sqrt((w.data**2).sum())

        if abs(t - temp) < 1e-10 or t <= eps:
            print('temp:', temp)
            print('t:', t)
            break
        w /= t
        print('{}, {}'.format(counter, t))
        reward_mtx = compute_reward_mtx(feat_mtx, w)

        # q-value iteration
        policy_matrix = value_iteration(reward_mtx, q_states,
                                               disc_rate, 0.001, 100)

        # q-learning
        # policy_matrix = q_learning(reward_mtx,
        #                            q_states,
        #                            disc_rate)
        policies.append(policy_matrix)
        mu_value = compute_policy_features_expectation(feat_mtx,
                                                       q_states,
                                                       policy_matrix,
                                                       disc_rate,
                                                       start_states,
                                                       term_states)
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


def value_iteration(reward_mtx, q_states, disc_rate, eps, max_reward):
    # q-value iteration

    # max_values = {s : (q_value, [a1, a2, ..., an])}
    # q_matrix = {(state, action): (row_index, state_prime}

    q_matrix = {}
    max_values = dict.fromkeys(list(q_states), (0, [0]))
    threshold = eps*(1-disc_rate)/disc_rate
    delta = threshold
    iteration = 1
    while delta >= threshold:
        delta = -1
        # print('iteration:', iteration)
        for state, actions in q_states.items():
            for action in actions:
                if action == -1:
                    # if action 'exit'
                    # reward = max_reward*exp(-((state[0]-16)/10)**2)

                    if (state, action) not in q_matrix:
                        reward = max_reward
                        # if 12 <= state[0] <= 16:
                        #     # inflate reward if in bar 12-16.
                        #     reward *= 3
                        q_matrix[(state, action)] = reward
                        max_values[state] = (reward, [action])
                    continue

                row_idx = q_states[state][action][0]
                state_prime = q_states[state][action][1]
                reward = reward_mtx[row_idx]
                opt_future_val = max_values[state_prime][0]
                new_q_value = reward + disc_rate * opt_future_val
                if (state, action) not in q_matrix:
                    diff = new_q_value
                    if diff < 0:
                        diff = -diff
                    q_matrix[(state, action)] = new_q_value
                else:
                    # if (state, action) in q_matrix
                    diff = new_q_value - q_matrix[(state, action)]
                    if diff < 0:
                        diff = -diff
                    q_matrix[(state, action)] = new_q_value

                # update max_values
                if max_values[state][0] < new_q_value:
                    max_values[state] = (new_q_value, [action])
                elif max_values[state][0] == new_q_value:
                    max_values[state][1].append(action)

                if diff > delta:
                    delta = diff
        iteration += 1
        # print('delta', delta)
    policy_matrix = {s: list(set(v[1])) for s, v in max_values.items()}
    return policy_matrix


def q_learning(reward_mtx, q_states, disc_rate, n_iter=50):
    # q-learning
    # use for loop over all actions. The size of states and actions is not
    # too large

    q_matrix = {(s, a): (0, 1) for s, acts in q_states.items() for a in acts}

    # should init max_values with random actions
    max_values = {s: (0, next(iter(a))) for s, a in q_states.items()}

    for k in range(0, n_iter):
        for state, actions in q_states.items():
            for action in actions:
                if action == -1:
                    # if action 'exit'
                    break
                row = q_states[state][action][0]
                reward = reward_mtx[row]
                state_prime = q_states[state][action][1]
                sample = reward + disc_rate * max_values[state_prime][0]
                n_visit = q_matrix[(state, action)][1]
                alpha = 1/n_visit
                new_q_value = alpha * sample + (1-alpha)*q_matrix[(state,
                                                                   action)][0]
                q_matrix[(state, action)] = (new_q_value, n_visit+1)

                # update max_values
                if max_values[state][0] < new_q_value:
                    max_values[state] = (new_q_value, action)
    policy_matrix = {s: choice(v[1]) for s, v in max_values.items()}
    return policy_matrix



def compute_expert_features_expectation(feat_mtx, q_states, disc_rate):
    trajectories = load_obj('TRAJECTORIES')
    expert_feat_exp = 0
    for trajectory in trajectories:
        sum_of_feat_exp = 0
        t = 0
        # from the first state to the penultimate state
        for state, action in zip(trajectory[::2], trajectory[1::2]):
            row = q_states[state][action][0]
            feat_exp = feat_mtx[row]
            discounted_feat_exp = disc_rate ** t * feat_exp
            sum_of_feat_exp += discounted_feat_exp
            t += 1
        expert_feat_exp += sum_of_feat_exp
    expert_feat_exp /= len(trajectories)
    return expert_feat_exp

def solve_lambdas(mu):
    """
    Solve qp problem that will return lambdas, the weights of linear
    combination of mu_0, mu_1, ..., mu_n which makes the linear combination
    the closest point to mu_expert.
    """
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
    return lambdas



if __name__ == '__main__':
    try:
        policies, mu = compute_policies(0.9999, 0.1)
        solve_lambdas(mu)
    except KeyboardInterrupt:
        pass

