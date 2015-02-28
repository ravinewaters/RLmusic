__author__ = 'redhat'

from common_methods import compute_next_state, load_obj, weighted_choice_b
import music21 as m
from random import choice


def mix_policies(policies, lambdas):
    idx = weighted_choice_b(lambdas)
    return policies[idx]


def generate_trajectory(start_state, term_states, policy_matrix):
    """
    Generate trajectory using start_state as the start state and
    policy_matrix as the policy until landing on term_states and action == -1.
    """
    state = start_state
    states = []
    counter = 0
    while True:
        states.append(state)
        action = choice(policy_matrix[state])
        if state in term_states and action == -1:
            break
        state = compute_next_state(state, action)
        counter += 1
        if counter == 30:
            break
    return states


def get_original_state(states, fignotes_dict, chords_dict):
    """
    returns the original states, e.g. (2, (60, 1.0), 'C', 4.0, 1.0, 60)
    """

    new_states = []
    for state in states:
        new_state = [state[0],
                     fignotes_dict[1][state[1]],
                     chords_dict[1][state[2]],
                     int(state[3] / 4.0),
                     int(state[4] / 4.0),
                     state[5]]
        new_states.append(tuple(new_state))
    return new_states


def translate_states_to_song(original_states, title='', composer=''):
    # use music21

    score = m.stream.Score()
    part = m.stream.Part()
    stream = m.stream.Stream()

    key_signature = m.key.Key('C')
    clef = m.clef.TrebleClef()
    common_time = m.meter.TimeSignature('4/4')
    instrument = m.instrument.Violin()

    score.append(m.metadata.Metadata())
    part.append(clef)
    part.append(key_signature)
    part.append(instrument)
    stream.append(common_time)

    score.metadata.title = title
    score.metadata.composer = composer

    for state in original_states:
        if state[-1] == -1:
            duration = state[1][1]
            r = m.note.Rest(quarterLength=duration)
            stream.append(r)
        else:
            pitches = state[1][::2]
            durations = state[1][1::2]
            for pitch, duration in zip(pitches, durations):
                n = m.note.Note(pitch, quarterLength=duration)
                stream.append(n)
    stream.makeMeasures(inPlace=True)
    part.append(stream)
    score.append(part)
    return score


def generate_score():
    policies = load_obj('POLICIES')
    lambdas = load_obj('LAMBDAS')
    start_states = load_obj('START_STATES')
    term_states = load_obj('TERM_STATES')
    fignotes_dict = load_obj('FIGNOTES_DICT')
    chords_dict = load_obj('CHORDS_DICT')

    policy = mix_policies(policies, lambdas)
    trajectory = generate_trajectory(choice(list(start_states)), term_states,
                                policy)
    song = get_original_state(trajectory, fignotes_dict, chords_dict)
    score = translate_states_to_song(song)
    score.show('musicxml')


if __name__ == '__main__':
    pass
