__author__ = 'redhat'

from common_methods import load_obj
from generate_features import compute_binary_features_expectation
from features_expectation import compute_policy_features_expectation
import numpy import linalg as LA
import random

def compute_policies(disc_rate, epsilon):
    all_states = load_obj('ALL_STATES')
    all_actions = load_obj('ALL_ACTIONS')
    start_states = load_obj('START_STATES')
    state_elem_size = load_obj('STATE_ELEM_SIZE')
    policy_matrix = generate_random_policy_matrix(all_states,
                                                  all_actions,
                                                  state_elem_size)
    expert_feat_exp = compute_expert_features_expectation(disc_rate)
    pi = []
    first = True
    counter = 1
    while True:
        if first:
            pi_value = compute_policy_features_expectation(policy_matrix,
                                                               disc_rate,
                                                               start_states,
                                                               1,)
            pi.append(pi_value)
            pi_bar = pi[counter-1]
            first = False
        else:
            pi_bar += projection
        w = expert_feat_exp - pi_bar
        t = LA.norm(w)
        if t <= epsilon:
            break
        policy_matrix = value_iteration_algorithm()
        pi_value = compute_policy_features_expectation(policy_matrix,
                                                               disc_rate,
                                                               start_states,
                                                               1,)
        pi.append(pi_value)
        counter += 1



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
    print(compute_expert_features_expectation(0.99))