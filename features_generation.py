__author__ = 'redhat'

from preprocess import BasePreprocessor
from scipy.sparse import csr_matrix
from scipy import io
from constants import CHORD_ROOT_TO_INT, DIR, PICKUP_ID
from numpy import uint8
from itertools import accumulate
import argparse
import shutil
import os

class FeaturesPreprocessor(BasePreprocessor):
    @staticmethod
    def parse_chord(chord):
        if len(chord) == 1:
            return chord
        chord = chord[:2]
        if chord[1] == '#' or chord[1] == 'b':
            return chord[:2]
        else:
            return chord[0]

    def compute_root_movement(self, state, action):
        # TROUBLE, how to parse chord?
        if action == -1:
            return 99
        if state[-1] == -1:
            return 100
        if action[-1] == -1:
            # next fig is rest
            return 101
        if state[2] == PICKUP_ID:
            # pickup, no chord
            return 102
        chord_root = self.chords_dict[1][state[2]]
        next_chord_root = self.chords_dict[1][action[1]]
        int_chord_root = CHORD_ROOT_TO_INT[self.parse_chord(chord_root)]
        int_next_chord_root = CHORD_ROOT_TO_INT[
            self.parse_chord(next_chord_root)
        ]
        return int_next_chord_root - int_chord_root

    @staticmethod
    def get_bar_number(state):
        return state[0]

    @staticmethod
    def get_current_figure_head_pitch(state):
        if state[-1] == -1:
            return 100
        return state[-1]

    @staticmethod
    def compute_figure_head_movement(state, action):
        if action == -1:
            # exit action
            return 99
        if state[-1] == -1:
            # current is rest
            return 100
        if action[-1] == -1:
            # next fig is rest
            return 101
        return action[-1] - state[-1]

    @staticmethod
    def get_current_beat(state):
        return state[3]

    @staticmethod
    def get_next_beat(state):
        return state[4] + state[3]

    @staticmethod
    def is_in_goal_state(state, action):
        if action == -1:
            return 1
        return 0

    @staticmethod
    def is_rest(state):
        if state[-1] == -1:
            return 1
        return 0

    def compute_features(self, state, action):
        # (root mvt,
        # current bar number
        # current fighead pitch,
        # fighead mvt,
        # get_current_beat
        # get_next_beat,
        # is_in_goal_state,
        # is_rest,
        # )

        feat_l = [
            self.compute_root_movement(state, action),
            self.get_bar_number(state),
            self.get_current_figure_head_pitch(state),
            self.compute_figure_head_movement(state, action),
            self.get_current_beat(state),
            self.get_next_beat(state),
            self.is_in_goal_state(state, action),
            self.is_rest(state),
        ]
        return feat_l

    def compute_proper_features(self, state, action, dictionaries, counters,
                                num_of_features):
        # use dictionary to get unique value of each coordinates so there is no
        # wasted coordinates in the binary features.
        feat = self.compute_features(state, action)
        for i in range(num_of_features):
            if feat[i] not in dictionaries[i]:
                dictionaries[i][feat[i]] = counters[i]
                counters[i] += 1
            feat[i] = dictionaries[i][feat[i]]
        return feat

    @staticmethod
    def compute_binary_features_expectation(cols, rows, row_idx, feat,
                                            l_elem_idx_marker):
        for i in range(len(feat)):
            idx = l_elem_idx_marker[i] + feat[i]
            cols.append(idx)
            rows.append(row_idx)

    def generate_features_expectation_mtx(self):
        """
        :return a sparse matrix called feat_mtx.
        """

        # map feature value to integer
        # e.g. {1: 0, 4:1, 100:2}
        # if this is not mapped, then the binary feature needs 100 coordinates,
        # otherwise only 3 coordinates is required.
        # the mapped feature is called proper feature
        # this will then be mapped to binary feature.
        #
        # temp_dict = {row_idx: feat}
        q_states = self.q_states
        num_of_features = 8
        dictionaries = [{} for _ in range(num_of_features)]
        counters = [0] * num_of_features
        temp_dict = {}

        # Get proper features and the size of each coordinate
        for state, actions in q_states.items():
            for action, value in actions.items():
                row_idx = value[0]
                feat = self.compute_proper_features(state,
                                                    action,
                                                    dictionaries,
                                                    counters,
                                                    num_of_features)
                temp_dict[row_idx] = feat

        coord_size = [len(d) for d in dictionaries]
        print('Size of coordinate elements:', coord_size[1:])
        l_elem_idx_marker = list(accumulate([0] + coord_size[:-1]))

        # use row_idx as the row number to store feat_exp into
        # use csr_matrix
        cols = []  # this will be a list of column numbers
        rows = []
        n_rows = -1  # get maximum row_idx value for csr_matrix construction
        for state, actions in q_states.items():
            for value in actions.values():
                row_idx = value[0]
                feat = temp_dict[row_idx]
                self.compute_binary_features_expectation(cols,
                                                         rows,
                                                         row_idx,
                                                         feat,
                                                         l_elem_idx_marker)
                if row_idx > n_rows:
                    n_rows = row_idx

        # create csr_matrix from data, rows and cols.
        data = [1] * len(cols)
        n_rows += 1  # include index 0
        n_cols = sum(coord_size)
        print('Number of state-action pairs:', n_rows)
        print('Binary features size:', n_cols)
        self.feat_mtx = csr_matrix((data, (rows, cols)), dtype=uint8,
                                   shape=(n_rows, n_cols))

        # save matrix to file
        io.savemat(DIR + 'FEATURES_EXPECTATION_MATRIX', {'mtx': self.feat_mtx})

    def run(self, corpus_dir='corpus/'):
        if os.path.exists(DIR):
            shutil.rmtree(DIR)

        if corpus_dir[-1] != '/':
            corpus_dir += '/'

        filenames = self.get_corpus(corpus_dir)
        l_states = [self.parse(filename) for filename in filenames]

        l_int_states = self.convert_each_elem_to_int(l_states)

        all_states = {state for states in l_int_states for state in states}
        all_actions = self.get_all_actions(all_states)

        self.get_trajectories(l_int_states)
        self.get_start_states()
        self.generate_all_possible_q_states(all_states, all_actions)
        self.generate_features_expectation_mtx()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Preprocessor module",
                                     usage="Specify the directory of "
                                           "the corpus")
    parser.add_argument('-d', '--dir', default='corpus/')
    args = parser.parse_args()
    preprocessor = FeaturesPreprocessor()
    preprocessor.run(args.dir)