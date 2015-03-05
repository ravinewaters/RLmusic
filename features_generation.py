__author__ = 'redhat'

from preprocess import BasePreprocessor
from scipy.sparse import csr_matrix
from scipy import io
from constants import CHORD_ROOT_TO_INT, DIR
from numpy import uint8
from common_methods import make_flat_list
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
        if state[2] == 1:
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

    def is_in_goal_state(self, state, action):
        if state in self.term_states and action == -1:
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
                                            coord_size):
        # modify cols and rows list
        # called after we know the size of each coordinate

        index = 0
        for i in range(len(feat)):
            index = index + coord_size[i]
            cols.append(index + feat[i])
            rows.append(row_idx)

    def generate_features_expectation_mtx(self):

        num_of_features = 8
        dictionaries = [{} for _ in range(num_of_features)]
        counters = [0] * num_of_features

        temp_dict = {}  # store proper features temporarily {row_idx: feat}
        q_states = self.q_states

        # this loop is to get proper features and the size of each coordinate
        for state, actions in q_states.items():
            for action, value in actions.items():
                row_idx = value[0]
                feat = self.compute_proper_features(state,
                                                    action,
                                                    dictionaries,
                                                    counters,
                                                    num_of_features)
                temp_dict[row_idx] = feat

        coord_size = [0] + [len(d) for d in dictionaries]
        print('Size of coordinate elements:', coord_size[1:])

        # use row_idx as the row number to store feat_exp into
        # use csr_matrix
        cols = []  # this will be a list of column
        rows = []
        n_rows = -1
        for state, actions in q_states.items():
            for value in actions.values():
                row_idx = value[0]
                feat = temp_dict[row_idx]
                self.compute_binary_features_expectation(cols,
                                                         rows,
                                                         row_idx,
                                                         feat,
                                                         coord_size)
                if row_idx > n_rows:
                    n_rows = row_idx

        # create csr_matrix from data, rows and cols.
        data = [1] * len(cols)
        n_rows += 1  # include index 0
        n_cols = sum(coord_size) + 1
        print('Number of state-action pairs:', n_rows)
        print('Binary features size:', n_cols)
        mtx = csr_matrix((data, (rows, cols)),
                         dtype=uint8,
                         shape=(n_rows, n_cols))

        # save matrix to file
        io.savemat(DIR + 'FEATURES_EXPECTATION_MATRIX', {'mtx': mtx})
        return mtx

    def run(self, corpus_dir='corpus/'):
        if os.path.exists(DIR):
            shutil.rmtree(DIR)

        if corpus_dir[-1] != '/':
            corpus_dir += '/'

        filenames = self.get_corpus(corpus_dir)
        list_of_song_states = [self.parse(filename) for filename in filenames]

        new_list_of_song_states, self.fignotes_dict, self.chords_dict \
            = self.convert_each_elem_to_int(list_of_song_states)

        all_states = set(make_flat_list(new_list_of_song_states))
        all_actions = self.get_all_actions(all_states)

        self.trajectories = self.get_trajectories(new_list_of_song_states)
        self.start_states = self.get_start_states()
        self.term_states = self.get_terminal_states()
        self.q_states = self.generate_all_possible_q_states(all_states,
                                                            all_actions)
        self.feat_mtx = self.generate_features_expectation_mtx()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Preprocessor module",
                                     usage="Specify the directory of "
                                           "the corpus")
    parser.add_argument('-d', '--dir', default='corpus/')
    args = parser.parse_args()
    preprocessor = FeaturesPreprocessor()
    preprocessor.run(args.dir)