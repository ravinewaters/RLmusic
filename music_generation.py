__author__ = 'redhat'

from common_methods import *
from features_expectation import choose_action_from_policy_matrix
import music21 as m


def generate_trajectory(start_state, term_states, all_actions,
                                policy_matrix):
    state = start_state
    state_size = load_obj('STATE_ELEM_SIZE')
    action_size = state_size[:2]
    states = []
    while state not in term_states:
        states.append(state)
        action = choose_action_from_policy_matrix(policy_matrix,
                                                  all_actions,
                                                  state,
                                                  state_size,
                                                  action_size)
        state = compute_next_state(state, action)
    states.append(state)  # append terminal state
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
        pitches = state[0][::2]
        durations = state[0][1::2]
        for pitch, duration in zip(pitches, durations):
            n = m.note.Note(pitch, quarterLength=duration)
            stream.append(n)
    stream.makeMeasures(inPlace=True)
    part.append(stream)
    score.append(part)
    return score

def convert_score_to_audio(score):
    pass


if __name__ == '__main__':
    trajectory = load_obj('TRAJECTORIES')[0][::2]
    fignotes_dict = load_obj('FIGNOTES_DICT')
    chords_dict = load_obj('CHORDS_DICT')
    song = get_original_state(trajectory, fignotes_dict, chords_dict)
    score = translate_states_to_song(song)
    score.show('musicxml')