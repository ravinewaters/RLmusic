__author__ = 'redhat'

from music21 import converter, note, harmony, stream
from common_methods import save_obj, compute_next_state
from constants import PICKUP_ID
import os


class BasePreprocessor():
    @staticmethod
    def parse(filename):
        """
        Idea:
        1. If current object is a chord or current note has beat 1.0 or a rest,
        2. save all the following notes,
        3. until meeting next chord or note that has beat 1.0.
        4. Pickup bar is handled separately
        5. A rest is considered a figure.

        state = (elem.measureNumber, tuple(fig_notes), fig_chord,
                 fig_start_at_beat, fig_duration, fighead)

        """
        states = []
        song = converter.parse(filename)
        elements = song.flat.getElementsByClass([harmony.ChordSymbol,
                                                 note.Rest,
                                                 note.Note])

        anacrusis = False
        first_measure = song.parts[0][1]

        # check anacrusis here instead of inside loop
        if first_measure.duration != first_measure.barDuration:
            anacrusis = True

            pickup = []
            pickup_fighead = None

            for elem in first_measure.notesAndRests:
                if elem.isNote:
                    if not pickup_fighead:
                        pickup_fighead = elem.midi
                        pickup_beat = elem.beat
                        pickup_duration = 0
                    pickup.append(elem.midi)

                elif elem.isRest and pickup_fighead:
                    pickup.append('rest')

                if pickup_fighead:
                    pickup.append(elem.quarterLength)
                    pickup_duration += elem.quarterLength

            states.append((elem.measureNumber,
                           tuple(pickup),
                           'pickup',
                           pickup_beat,
                           pickup_duration,
                           pickup_fighead))

        last_item = False
        for i in range(len(elements)):
            elem = elements[i]
            if elem.measureNumber == 1 and anacrusis is True:
                continue

            prev_elem = elements[i-1]
            try:
                next_elem = elements[i+1]
            except stream.StreamException:
                # at the last iteration, get the next item from the original
                # Measure
                next_elem = elem.next()
                last_item = True

            if elem.isChord:
                fig_chord = elem.figure  # get chord's name

            elif elem.isNote:
                if prev_elem.isChord or prev_elem.isRest or elem.beat == 1.0:
                    fighead = elem.midi
                    fig_start_at_beat = elem.beat
                    fig_notes = []
                    fig_duration = 0

                fig_notes.append(elem.midi)
                fig_notes.append(elem.quarterLength)
                fig_duration += elem.quarterLength

                # Wrap up figure if we encounter Rest or Chord or Final
                # Barline (next_elem has not attribute 'pitch') or new bar
                # (beat == 1.0)
                if not hasattr(next_elem, 'pitch') or next_elem.beat == 1.0:
                    figure = (elem.measureNumber,  #  0
                              tuple(fig_notes),    #  1
                              fig_chord,           #  2
                              fig_start_at_beat,   #  3
                              fig_duration,        #  4
                              fighead)             #  5
                    states.append(figure)

            # If rest is the last_item, ignore it.
            elif elem.isRest and not last_item:
                states.append((elem.measureNumber,
                               ('rest', elem.quarterLength),
                               'rest',
                               elem.beat,
                               elem.quarterLength,
                               -1))
        return states

    @staticmethod
    def get_corpus(corpus_dir):
        """
        Return a list of filenames in the corpus.
        """
        filenames = []
        for f in os.listdir(corpus_dir):
            if '.xml' in f and os.path.isfile(corpus_dir + f):
                filenames.append(corpus_dir + f)
        return filenames

    def convert_each_elem_to_int(self, l_states):
        """
        returns states that are tuple of nonnegative integers, e.g.
        (1,1,1,1,2) instead of ((0, 1.0), 'C', 1.0, 1.0, 0)
        """

        # (dict_elem_to_int, dict_int_to_elem)
        fignotes_dict = ({}, {})
        chords_dict = ({}, {})
        fignotes_counter = 1
        chords_counter = 1

        new_all_states = []
        for states in l_states:
            new_states = []
            for state in states:
                fignotes = state[1]
                chord = state[2]

                if fignotes in fignotes_dict[0]:
                    fignotes_int = fignotes_dict[0][fignotes]
                else:
                    fignotes_dict[0][fignotes] = fignotes_counter
                    fignotes_dict[1][fignotes_counter] = fignotes
                    fignotes_int = fignotes_counter
                    fignotes_counter += 1

                if chord in chords_dict[0]:
                    chord_int = chords_dict[0][chord]
                else:
                    if chord != 'pickup':
                        chords_dict[0][chord] = chords_counter
                        chords_dict[1][chords_counter] = chord
                        chord_int = chords_counter
                        chords_counter += 1
                    else:
                        # if pickup
                        chords_dict[0][chord] = PICKUP_ID
                        chords_dict[1][PICKUP_ID] = chord
                        chord_int = PICKUP_ID

                new_state = (state[0],
                             fignotes_int,
                             chord_int,
                             int(state[3] * 4),
                             int(state[4] * 4),
                             state[5])
                new_states.append(new_state)
            new_all_states.append(new_states)

        self.fignotes_dict = fignotes_dict
        self.chords_dict = chords_dict

        save_obj(fignotes_dict, 'FIGNOTES_DICT')
        save_obj(chords_dict, 'CHORDS_DICT')

        return new_all_states

    @staticmethod
    def compute_action(s_prime):
        """find a that makes s transition to s_prime"""

        action = s_prime[1:3] + s_prime[-2:]
        return action

    def get_trajectories(self, list_of_song_states):
        trajectories = []
        for states in list_of_song_states:
            trajectory = []
            first = True
            for state in states:
                if first:
                    trajectory.append(state)
                    first = False
                    continue
                action = self.compute_action(state)
                trajectory.append(action)
                trajectory.append(state)
            trajectory.append(-1)  # append exit action to trajectory
            trajectories.append(trajectory)
        self.trajectories = trajectories
        save_obj(trajectories, 'TRAJECTORIES')

    def get_start_states(self):
        self.start_states = {trajectory[0] for trajectory in self.trajectories}
        save_obj(self.start_states, 'START_STATES')

    def get_all_actions(self, all_states):
        # actions = (seq_of_notes, chord, duration, figurehead)
        all_actions = []
        for state in all_states:
            chord = state[2]
            if chord == PICKUP_ID:
                continue
            else:
                all_actions.append(state[1:3] + state[-2:])
        return set(all_actions)

    def generate_all_possible_q_states(self, all_states, all_actions):
        """
        Return q_states = {s : {a: (row_idx, s')}}

        row_idx is a row number in which we store feat_exp of corresponding
        state, action into.

        s is state
        a is possible state for s
        s' is the next state from s after having chosen action a

        A terminal state s have action a = -1 and s' = - 1.
        """

        term_states = {trajectory[-2] for trajectory in self.trajectories}
        row_idx = 0
        q_states = {}
        for state in all_states:
            if state in term_states:
                # Give 'exit' action to terminal states
                q_states[state] = {-1: (row_idx, -1)}
                row_idx += 1

            for action in all_actions:
                next_state = compute_next_state(state, action)
                if next_state in all_states:
                    try:
                        q_states[state][action] = (row_idx, next_state)
                    except KeyError:
                        q_states[state] = {action: (row_idx, next_state)}
                    row_idx += 1
        self.q_states = q_states
        save_obj(q_states, 'Q_STATES')

if __name__ == "__main__":
    print("This module can't be use directly")
    pass