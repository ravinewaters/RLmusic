__author__ = 'redhat'

from common_methods import load_obj, DIR, save_obj
from math import sqrt
from scipy import sparse, io
from cvxopt import matrix, spmatrix, solvers
from random import choice
import os


class ALAlgorithm():
    def __init__(self, disc_rate, eps, preprocessor=None):
        self.disc_rate = disc_rate
        self.eps = eps

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

    def compute_policies(self):
        # max_reward = 6
        # value_error = 0.01
        print('\ndisc_rate', self.disc_rate)
        print('eps:', self.eps)
        # print('value_error:', value_error)
        # print('terminal states reward:', max_reward)

        # bind to name for faster access
        eps = self.eps
        feat_mtx = self.feat_mtx
        compute_policy_features_expectation = self.compute_policy_features_expectation
        compute_projection = self.compute_projection
        value_iteration = self.value_iteration

        try:
            # Load saved computation state
            temp = io.loadmat(DIR + 'TEMP')
            policies = load_obj('TEMP_POLICIES')
            mu_expert = temp['mu_expert']
            mu = temp['mu'].tolist()[0]
            mu_bar = temp['mu_bar']
            counter = temp['counter'][0][0]
        except FileNotFoundError:
            policy_matrix = self.generate_random_policy_matrix()
            policies = [policy_matrix]
            mu_expert = self.compute_expert_features_expectation()
            mu = []
            counter = 1

        print('\n', 'counter, t')
        t = 0
        while counter <= 30:
            if counter == 1:
                mu_value = compute_policy_features_expectation(policy_matrix)
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
            reward_mtx = (feat_mtx * w.T).data

            # q-value iteration
            policy_matrix = value_iteration(reward_mtx, 0.001, 100)

            # q-learning
            # policy_matrix = q_learning(reward_mtx,
            #                            q_states,
            #                            disc_rate)
            policies.append(policy_matrix)
            mu_value = compute_policy_features_expectation(policy_matrix)
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
        self.policies = policies
        self.mu = mu

    @staticmethod
    def compute_projection(mu_bar, mu, mu_expert):
        mu_mu_bar_distance = mu - mu_bar
        mu_bar_mu_expert_distance = mu_expert - mu_bar
        numerator = mu_mu_bar_distance.dot(mu_bar_mu_expert_distance.T)[0, 0]
        denominator = mu_mu_bar_distance.dot(mu_mu_bar_distance.T)[0, 0]
        return numerator/denominator * mu_mu_bar_distance

    def value_iteration(self, reward_mtx, eps, max_reward):
        # q-value iteration

        # max_values = {s : (q_value, [a1, a2, ..., an])}
        # q_matrix = {(state, action): (row_index, state_prime}
        disc_rate = self.disc_rate
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


    def q_learning(self, reward_mtx, n_iter=50):
        # q-learning
        # use for loop over all actions. The size of states and actions is not
        # too large

        q_states = self.q_states
        disc_rate = self.disc_rate

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

    def compute_policy_features_expectation(self, policy_matrix):
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

        feat_mtx = self.feat_mtx
        q_states = self.q_states
        disc_rate = self.disc_rate
        start_states = self.start_states
        term_states = self.term_states

        mean_feat_exp = 0
        for state in start_states:
            sum_of_feat_exp = 0
            t = 0
            while True:
                action = choice(policy_matrix[state])
                row = q_states[state][action][0]
                feat_exp = feat_mtx[row]
                discounted_feat_exp = disc_rate ** t * feat_exp
                sum_of_feat_exp += discounted_feat_exp

                if state in term_states and action == -1:
                    break
                elif discounted_feat_exp.sum() <= 1e-10:
                    break

                # next state
                state = q_states[state][action][1]
                t += 1
            mean_feat_exp += sum_of_feat_exp

        return mean_feat_exp/len(start_states)


    def compute_expert_features_expectation(self):
        disc_rate = self.disc_rate
        feat_mtx = self.feat_mtx
        q_states = self.q_states
        trajectories = self.trajectories

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

    def solve_lambdas(self):
        """
        Solve qp problem that will return lambdas, the weights of linear
        combination of mu_0, mu_1, ..., mu_n which makes the linear combination
        the closest point to mu_expert.
        """
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
        B = sparse.vstack(self.mu)
        P = (B * B.T).tocoo()
        P = spmatrix(P.data, P.row, P.col)
        self.lambdas = list(solvers.qp(2*P, q, G, h, A, b)['x'])[1:]
        save_obj(self.lambdas, 'LAMBDAS')

if __name__ == '__main__':
    algo = ALAlgorithm(0.95, 1)
    algo.compute_policies()
    algo.solve_lambdas()
    print(algo.policies)
    print(algo.lambdas)


