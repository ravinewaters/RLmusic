__author__ = 'redhat'

from random import choice
from scipy import sparse, io
from numpy import float64, sqrt, array
from cvxopt import matrix, spmatrix, solvers
from common_methods import load_obj, DIR, save_obj
import os
import argparse
import itertools


class ALAlgorithm():
    def __init__(self, preprocessor=None):
        if preprocessor is not None:
            self.start_states = preprocessor.start_states
            self.q_states = preprocessor.q_states
            self.feat_mtx = preprocessor.feat_mtx
            self.trajectories = preprocessor.trajectories
        else:
            self.start_states = load_obj('START_STATES')
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
            policy_matrix = self.generate_initial_policy_matrix()
            policies = [policy_matrix]
            mu_expert = self.compute_expert_features_expectation(disc_rate)
            mu = []
            counter = 1

        print('\n', 'counter, t')
        t = 0
        while counter <= 50:
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

            print('{}, {}'.format(counter, t))

            reward_mtx = feat_mtx * w.T
            policy_matrix = value_iteration(reward_mtx, disc_rate, max_reward)
            policies.append(policy_matrix)
            mu_value = compute_policy_features_expectation(policy_matrix,
                                                           disc_rate)
            mu.append(mu_value)
            counter += 1

            io.savemat(DIR + 'TEMP', {'mu_expert': mu_expert, 'mu': mu,
                                      'mu_bar': mu_bar, 'counter': counter})
            save_obj(policies, 'TEMP_POLICIES')

        mu = [mu_expert] + mu  # stack at the beginning.
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
        # q_states = {s : {a: (row_idx, s')}}
        # q_matrix = {(state, action): q-value}

        # use try-except when accessing max_values
        # if doesn't exist return 0

        eps = 0.001
        q_states = self.q_states
        q_matrix = {}
        max_values = {}
        threshold = 2*eps*(1-disc_rate)/disc_rate
        delta = threshold
        while delta >= threshold:
            delta = -1
            for state, actions in q_states.items():
                for action in actions:
                    row_idx = q_states[state][action][0]
                    reward = reward_mtx[row_idx]
                    opt_future_val = 0

                    if action != -1:
                        state_prime = q_states[state][action][1]
                        try:
                            opt_future_val = max_values[state_prime][0]
                        except KeyError:
                            pass
                    else:
                        reward += max_reward

                    new_q_value = reward + disc_rate * opt_future_val

                    if (state, action) in q_matrix:
                        diff = new_q_value - q_matrix[(state, action)]
                    else:
                        diff = new_q_value
                    q_matrix[(state, action)] = new_q_value

                    try:
                        # update max_values
                        if max_values[state][0] < new_q_value:
                            max_values[state] = (new_q_value, [action])
                        elif max_values[state][0] == new_q_value:
                            max_values[state][1].append(action)
                    except KeyError:
                        max_values[state] = (new_q_value, [action])

                    if diff < 0:
                        diff = -diff

                    if diff > delta:
                        delta = diff
        policy_matrix = {s: tuple(set(v[1])) for s, v in max_values.items()}
        return policy_matrix

    def generate_initial_policy_matrix(self):
        """
        Generate a policy table = {s: [a_1, a_2, ..., a_k]}, where
        a_1, a_2, ..., a_k are valid actions for state s.
        """
        policy_matrix = {s: tuple(v) for s, v in self.q_states.items()}
        return policy_matrix

    def compute_policy_features_expectation(self, policy_matrix, disc_rate):
        """
        Generate a set of trajectories starting from state in start states
        until picking 'exit' action.
        The number of trajectories is 7 * number of state in start_states.
        Then, we average the discounted features vector over all
        trajectories in the set to get an estimate of features expectation
        of the policy.
        """

        feat_mtx = self.feat_mtx
        q_states = self.q_states
        start_states = self.start_states
        row = 0
        row_ind = []
        col_ind = []
        data = []
        for _ in itertools.repeat(None, 7):
            for state in start_states:
                t = 0
                while True:
                    action = choice(policy_matrix[state])
                    row_idx = q_states[state][action][0]
                    row_ind.append(row)
                    col_ind.append(row_idx)
                    data.append(disc_rate ** t)
                    if action == -1:
                        break
                    state = q_states[state][action][1]  # next state
                    t += 1
                row += 1
        discount_mtx = sparse.csr_matrix((data, (row_ind, col_ind)),
                                         shape=(row, feat_mtx.shape[0]),
                                         dtype=float64)
        return (discount_mtx * feat_mtx).mean(0).A1

    def compute_expert_features_expectation(self, disc_rate):
        """
        Similar to compute_policy_features_expectation but we are given a
        fixed number of trajectories which are expert's.
        """
        feat_mtx = self.feat_mtx
        q_states = self.q_states
        trajectories = self.trajectories

        row = 0
        row_ind = []
        col_ind = []
        data = []
        for trajectory in trajectories:
            t = 0
            # zip states and actions
            for state, action in zip(trajectory[::2], trajectory[1::2]):
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

        # B = (n+1) x 1 array of arrays
        # P = (n+1) x (n+1) matrix
        # q = (n+1) x 1 zero-matrix
        # G = n x (n+1) matrix
        # h = 2 x 1 matrix
        # A = 2 x (n+1) matrix
        # b = 2 x 1 matrix

        solvers.options['show_progress'] = False
        n = len(self.mu) - 1
        A_data = [1]*(n+1)
        A_rows = [0] + [1]*n
        A_cols = range(n+1)
        A = spmatrix(A_data, A_rows, A_cols)
        b = matrix([1.0, 1.0])
        G_data = [-1]*n
        G_rows = range(0, n)
        G_cols = range(1, n+1)
        G_size = (n, n+1)
        G = spmatrix(G_data, G_rows, G_cols, G_size)
        h = matrix([0.0]*n)
        q = matrix([0.0]*(n+1))
        B = array(self.mu)
        P = matrix(B.dot(B.T))
        self.lambdas = list(solvers.qp(P, q, G, h, A, b)['x'])[1:]
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
    parser.add_argument('-mr', '--max_reward', default=100, type=float)
    args = parser.parse_args()

    alg = ALAlgorithm()
    alg.run(args.disc_rate, args.eps, args.max_reward)


