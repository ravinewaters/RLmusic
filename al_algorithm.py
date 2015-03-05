__author__ = 'redhat'

from numpy import float64, sqrt, vstack
from random import choice
from scipy import sparse, io
from cvxopt import matrix, spmatrix, solvers
from common_methods import load_obj, DIR, save_obj
import os
import argparse


class ALAlgorithm():
    def __init__(self, preprocessor=None):

        if preprocessor is not None:
            self.start_states = preprocessor.start_states
            self.term_states = preprocessor.term_states
            self.q_states = preprocessor.q_states
            self.feat_mtx = preprocessor.feat_mtx
            self.trajectories = preprocessor.trajectories
        else:
            self.start_states = load_obj('START_STATES')
            self.term_states = load_obj('TERM_STATES')
            self.q_states = load_obj('Q_STATES')
            self.feat_mtx = io.loadmat(DIR + 'FEATURES_EXPECTATION_MATRIX')[
                'mtx']
            self.trajectories = load_obj('TRAJECTORIES')

    def compute_policies(self, disc_rate=0.95,
                         al_error_tolerance=1,
                         max_reward=100):
        # bind to name for faster access
        feat_mtx = self.feat_mtx
        compute_policy_features_expectation = self.compute_policy_features_expectation
        compute_projection = self.compute_projection
        value_iteration = self.value_iteration

        print('\ndisc_rate', disc_rate)
        print('al_error_tolerance:', al_error_tolerance)
        print('max_reward:', max_reward)

        # check for bug
        # the rate of decrease of t is very low after only few iterations

        try:
            # Load saved computation state
            temp = io.loadmat(DIR + 'TEMP')
            policies = load_obj('TEMP_POLICIES')
            mu_expert = temp['mu_expert'].ravel()
            mu = temp['mu'].tolist()
            mu_bar = temp['mu_bar'].ravel()
            counter = temp['counter'][0][0]
        except FileNotFoundError:
            policy_matrix = self.generate_random_policy_matrix()
            policies = [policy_matrix]
            mu_expert = self.compute_expert_features_expectation(disc_rate)
            mu = []
            counter = 1

        print('\n', 'counter, t')
        t = 0
        while counter <= 30:
            if counter == 1:
                mu_value = compute_policy_features_expectation(policy_matrix,
                                                               disc_rate)
                mu.append(mu_value)
                mu_bar = mu[counter-1]  # mu_bar[0] = mu[0]
            else:  # counter >= 2
                mu_bar += compute_projection(mu_bar, mu[counter-1], mu_expert)

            w = mu_expert - mu_bar
            temp = t
            t = sqrt(w.dot(w.T))

            if abs(t - temp) < 1e-10 or t <= al_error_tolerance:
                print('temp:', temp)
                print('t:', t)
                break

            w /= t
            print('{}, {}'.format(counter, t))
            reward_mtx = (feat_mtx * w.T)

            # q-value iteration
            policy_matrix = value_iteration(reward_mtx, disc_rate, max_reward)

            # q-learning
            # policy_matrix = q_learning(reward_mtx,
            #                            disc_rate,
            #                            100)
            policies.append(policy_matrix)
            mu_value = compute_policy_features_expectation(policy_matrix,
                                                           disc_rate)
            mu.append(mu_value)
            counter += 1

            # save for later computation
            io.savemat(DIR + 'TEMP', {'mu_expert': mu_expert, 'mu': mu,
                                      'mu_bar': mu_bar, 'counter': counter})
            save_obj(policies, 'TEMP_POLICIES')

        mu = [mu_expert] + mu
        save_obj(policies, 'POLICIES')
        save_obj(mu, 'MU')

        if os.path.exists(DIR + 'TEMP.mat'):
            os.remove(DIR + 'TEMP.mat')
            os.remove(DIR + 'TEMP_POLICIES.pkl')

        self.policies = policies
        self.mu = mu

    @staticmethod
    def compute_projection(mu_bar, mu, mu_expert):
        mu_mu_bar_distance = mu - mu_bar
        mu_bar_mu_expert_distance = mu_expert - mu_bar
        numerator = mu_mu_bar_distance.dot(mu_bar_mu_expert_distance.T)
        denominator = mu_mu_bar_distance.dot(mu_mu_bar_distance.T)
        return numerator/denominator * mu_mu_bar_distance

    def value_iteration(self, reward_mtx, disc_rate, max_reward):
        # q-value iteration

        # max_values = {s : (q_value, [a1, a2, ..., an])}
        # q_matrix = {(state, action): (row_index, next_state}

        eps = 0.001
        q_states = self.q_states
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
                        # change reward to:
                        # reward for (s,-1) is reward_mtx[row_idx] + max_reward
                        if (state, action) not in q_matrix:
                            reward = max_reward
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

    def q_learning(self, reward_mtx, disc_rate, n_iter=50):
        # q-learning
        # use for loop over all actions. The size of states and actions is not
        # too large

        q_states = self.q_states

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
                    new_q_value = alpha * sample + \
                                  (1-alpha)*q_matrix[(state, action)][0]
                    q_matrix[(state, action)] = (new_q_value, n_visit+1)

                    # update max_values
                    if max_values[state][0] < new_q_value:
                        max_values[state] = (new_q_value, action)
        policy_matrix = {s: choice(v[1]) for s, v in max_values.items()}
        return policy_matrix


    def generate_random_policy_matrix(self):
        # generate matrix of 0-1 value with size:
        # rows = # states
        # cols = # actions
        # Use dictionary not matrix.
        # not stochastic
        # Should add stochastic policy to the matrix.

        policy_matrix = {s: list(v) for s, v in self.q_states.items()}
        # deterministic action
        return policy_matrix

    def compute_policy_features_expectation(self, policy_matrix, disc_rate):
        # Walk through the states and
        # actions. The actions are gotten by choosing randomly according to the
        # policy matrix. We start from a given start_state and stop when
        # reaching a terminal state or the features expectation is very small
        # because of the discount factor.
        # We generate n_iter trajectories and find the average of the sum of
        # discounted feature expectation. If we have a deterministic policy we
        # can just set the n_iter to be 1.

        # policy_matrix:
        # {state: ((a1, .05), (a2, .1), (a3, .85))}
        # row_ind, col_ind, data to construct discount_mtx

        feat_mtx = self.feat_mtx
        q_states = self.q_states
        start_states = self.start_states
        term_states = self.term_states

        row = 0
        row_ind = []
        col_ind = []
        data = []
        for state in start_states:
            t = 0
            while True:
                # can be vectorized. current approach is slow
                # 2 vectors
                # discounts = [disc_rate**t for t=1:n]
                # row_idxes = [row_idx]
                # then slice the feat_mtx using row_idxes
                # the dot product the slice with discounts
                action = choice(policy_matrix[state])
                row_idx = q_states[state][action][0]
                row_ind.append(row)
                col_ind.append(row_idx)
                data.append(disc_rate ** t)

                if state in term_states and action == -1:
                    break

                # next state
                state = q_states[state][action][1]
                t += 1
            row += 1

        discount_mtx = sparse.csr_matrix((data, (row_ind, col_ind)),
                                         shape=(row, feat_mtx.shape[0]),
                                         dtype=float64)

        # mean_feat_exp should be an array or csr_mtx.
        # how to multiply 2 sparse matrix and take its mean?
        return (discount_mtx * feat_mtx).mean(0).A1

    def compute_expert_features_expectation(self, disc_rate):
        feat_mtx = self.feat_mtx
        q_states = self.q_states
        trajectories = self.trajectories

        row = 0
        row_ind = []
        col_ind = []
        data = []
        for trajectory in trajectories:
            t = 0
            # from the first state to the penultimate state
            for state, action in zip(trajectory[::2], trajectory[1::2]):
                # can be vectorized as well, like the
                # compute_policy_features_expectation
                row_idx = q_states[state][action][0]
                row_ind.append(row)
                col_ind.append(row_idx)
                data.append(disc_rate ** t)
                t += 1
            row += 1
        discount_mtx = sparse.csr_matrix((data, (row_ind, col_ind)),
                                         shape=(row, feat_mtx.shape[0]),
                                         dtype=float64)
        return (discount_mtx * feat_mtx).mean(0).A1

    def solve_lambdas(self):
        """
        Solve qp problem that will return lambdas, the weights of linear
        combination of mu_0, mu_1, ..., mu_n which makes the linear combination
        the closest point to mu_expert.
        """

        # check once again whether the matrix P, q, G, h, A, b is already
        # correct

        solvers.options['show_progress'] = False
        n = len(self.mu) - 1
        A_data = [1]*(n+1)
        A_rows = [0] + [1]*n
        A_cols = range(n+1)
        A = spmatrix(A_data, A_rows, A_cols)
        b = matrix([1.0, 1.0])
        G = spmatrix([-1]*n, range(0, n), range(1, n+1), (n, n+1))
        h = matrix([0.0]*n)
        q = matrix([0.0]*(n+1))
        B = vstack(self.mu)
        P = matrix(B.dot(B.T))
        self.lambdas = list(solvers.qp(2*P, q, G, h, A, b)['x'])[1:]
        save_obj(self.lambdas, 'LAMBDAS')

    def run(self, disc_rate, eps, max_reward):
        self.compute_policies(disc_rate, eps, max_reward)
        self.solve_lambdas()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Apprenticeship Learning "
                                          "Algorithm module",
                                     usage="Specify the discount rate, "
                                           "eps and max_reward for the "
                                           "learning algorithm.")
    parser.add_argument('-dr', '--disc_rate', default=0.95, type=float)
    parser.add_argument('--eps', default=1, type=float)
    parser.add_argument('-mr', '--max_reward', default=1000, type=float)
    args = parser.parse_args()

    alg = ALAlgorithm()
    alg.run(args.disc_rate, args.eps, args.max_reward)


