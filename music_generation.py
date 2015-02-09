__author__ = 'redhat'

from common_methods import *
import music21 as m
from random import choice


def generate_trajectory(start_state, term_states, policy_matrix):
    state = start_state
    states = []
    counter = 0
    while True:
        states.append(state)
        # should use weighted_choice
        action = policy_matrix[state][0][0]
        if state in term_states and action == -1:
            break
        state = compute_next_state(state, action)
        counter += 1
        if counter == 30:
            break
    save_obj(states, 'GENERATED_SEQUENCE_OF_STATES')
    return states


def get_original_state(states, fignotes_dict, chords_dict):
    """
    returns states that are tuple of nonnegative integers, e.g.
    (1,1,1,1,2) instead of ((0, 1.0), 'C', 1.0, 1.0, 0)
    """

    new_states = []
    for state in states:
        new_state = [fignotes_dict[1][state[0]],
                     chords_dict[1][state[1]],
                     int(state[2] / 4.0),
                     int(state[3] / 4.0),
                     state[4]]
        new_states.append(tuple(new_state))
    return new_states


def translate_states_to_song(original_states, title='', composer=''):
    # use music21

    score = m.stream.Score()
    part = m.stream.Part()
    stream = m.stream.Stream()

    common_time = m.meter.TimeSignature('4/4')
    instrument = m.instrument.Violin()

    score.append(m.metadata.Metadata())
    part.append(instrument)
    stream.append(common_time)

    score.metadata.title = title
    score.metadata.composer = composer

    for state in original_states:
        if state[-1] == -1:
            duration = state[0][1]
            r = m.note.Rest(quarterLength=duration)
            stream.append(r)
        else:
            pitches = state[0][::2]
            durations = state[0][1::2]
            for pitch, duration in zip(pitches, durations):
                n = m.note.Note(pitch, quarterLength=duration)
                stream.append(n)
    stream.makeMeasures(inPlace=True)
    part.append(stream)
    score.append(part)
    return score


if __name__ == '__main__':
    policies = load_obj('POLICIES')
    start_states = load_obj('START_STATES')
    term_states = load_obj('TERM_STATES')
    index = load_obj('OPTIMAL_POLICY_INDEX')
    fignotes_dict = load_obj('FIGNOTES_DICT')
    chords_dict = load_obj('CHORDS_DICT')
    policy = policies[index]
    trajectory = generate_trajectory(choice(list(start_states)), term_states,
                                policy)
    song = get_original_state(trajectory, fignotes_dict, chords_dict)
    score = translate_states_to_song(song)
    score.show('musicxml')
