__author__ = 'redhat'

from music21 import converter, note, harmony, stream
from common_methods import save_obj, compute_next_state
import os


class BasePreprocessor():
    @staticmethod
    def parse(filename):
        """
        Idea:
        1. When it's a chord, the next note is a figurehead
        2. Continue acquiring all notes that falls into the same figure, until
        3. meeting the next chord, which implies new figure
        4. Pickup bar and the last figure are handled separately

        notes:
        1. In 4/4 beat occurs every quarter.

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

            states.append((elem.measureNumber, tuple(pickup), '0', pickup_beat,
                           pickup_duration, pickup_fighead))

        last_item = False
        for i in range(len(elements)):
            prev_elem = elements[i-1]
            elem = elements[i]
            try:
                next_elem = elements[i+1]
            except stream.StreamException:
                # at the last iteration, look for next item in the original measure
                next_elem = elem.next()
                last_item = True

            # Skip if anacrusis
            if elem.measureNumber == 1 and anacrusis:
                continue

            if elem.isChord:
                # get chord's name
                fig_chord = elem.figure

            elif elem.isNote:
                if prev_elem.isChord or prev_elem.isRest or elem.beat == 1.0:
                    fighead = elem.midi
                    fig_start_at_beat = elem.beat
                    fig_notes = []
                    fig_duration = 0

                fig_notes.append(elem.midi)
                fig_notes.append(elem.quarterLength)
                fig_duration += elem.quarterLength

                # Wrap up figure if we encounter Rest or Chord or new bar (beat ==
                # 1.0) or Final Barline
                if not hasattr(next_elem, 'pitch') or next_elem.beat == 1.0:
                    figure = (elem.measureNumber,  #  0
                              tuple(fig_notes),    #  1
                              fig_chord,           #  2
                              fig_start_at_beat,   #  3
                              fig_duration,        #  4
                              fighead)             #  5
                    states.append(figure)

            elif elem.isRest and not last_item:
                # elem is a rest
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

    @staticmethod
    def convert_each_elem_to_int(list_of_song_states):
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
        for states in list_of_song_states:
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
                    if chord != '0':
                        chords_dict[0][chord] = chords_counter
                        chords_dict[1][chords_counter] = chord
                        chord_int = chords_counter
                        chords_counter += 1
                    else:
                        # if pickup
                        chords_dict[0][chord] = 0
                        chords_dict[1][0] = chord

                new_state = (state[0],
                             fignotes_int,
                             chord_int,
                             int(state[3] * 4),
                             int(state[4] * 4),
                             state[5])
                new_states.append(new_state)
            new_all_states.append(new_states)

        save_obj(fignotes_dict, 'FIGNOTES_DICT')
        save_obj(chords_dict, 'CHORDS_DICT')
        return new_all_states, fignotes_dict, chords_dict

    @staticmethod
    def compute_action(s_prime):
        # find a that makes fs transition to s_prime
        # no bar in action
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
            trajectory.append(-1)  # exit action for terminal state
            trajectories.append(trajectory)
        save_obj(trajectories, 'TRAJECTORIES')
        return trajectories

    def get_terminal_states(self):
        terminal_states = []
        for trajectory in self.trajectories:
            terminal_states.append(trajectory[-2])
        return set(terminal_states)


    def get_start_states(self):
        start_states = []
        for trajectory in self.trajectories:
            start_states.append(trajectory[0])
        save_obj(start_states, 'START_STATES')
        return set(start_states)


    def get_all_actions(self, all_states):
        all_actions = []
        for state in all_states:
            chord = state[2]
            if chord == 0:
                # '0': 0 < 'A': 1, pickup will be assigned to 0
                continue
            else:
                all_actions.append(state[1:3] + state[-2:])
        return set(all_actions)


    def generate_all_possible_q_states(self, all_states, all_actions,
                                       term_states):
        """
        Return q_states = {s : {a: (row_idx, s')}}

        row_idx is a row number in which we store feat_exp of corresponding
        state, action into.

        s is state
        a is possible state for s
        s' is the next state from s after having chosen action a

        A terminal state s have action a = -1 and s' = - 1.
        """

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
        save_obj(q_states, 'Q_STATES')
        return q_states

if __name__ == "__main__":
    print("This module can't be use directly")
    pass