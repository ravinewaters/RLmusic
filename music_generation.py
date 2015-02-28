__author__ = 'redhat'

from common_methods import compute_next_state, load_obj, weighted_choice_b, \
    make_dir_when_not_exist
from random import choice
import subprocess
import music21
import sys



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

    score = music21.stream.Score()
    part = music21.stream.Part()
    stream = music21.stream.Stream()

    key_signature = music21.key.Key('C')
    clef = music21.clef.TrebleClef()
    common_time = music21.meter.TimeSignature('4/4')
    instrument = music21.instrument.Piano()

    score.append(music21.metadata.Metadata())
    part.append(clef)
    part.append(key_signature)
    part.append(instrument)
    stream.append(common_time)

    score.metadata.title = title
    score.metadata.composer = composer

    for state in original_states:
        if state[-1] == -1:
            duration = state[1][1]
            r = music21.note.Rest(quarterLength=duration)
            stream.append(r)
        else:
            pitches = state[1][::2]
            durations = state[1][1::2]
            for pitch, duration in zip(pitches, durations):
                n = music21.note.Note(pitch, quarterLength=duration)
                stream.append(n)
    stream.makeMeasures(inPlace=True)
    part.append(stream)
    score.append(part)
    return score


def convert_midi_to_mp3_file(output_path, soundfont_path):
    # requires fluidsynth and lame

    wavfile = output_path+'.wav'
    midifile = output_path+'.mid'
    midi_to_wav_cmd = "fluidsynth -F {} {} {}".format(wavfile,
                                                      soundfont_path,
                                                      midifile)
    wav_to_mp3_cmd = "lame {} -V 9".format(wavfile)
    rm_wavfile_cmd = "rm {}".format(wavfile)
    try:
        subprocess.call(midi_to_wav_cmd.split(' '))
        subprocess.call(wav_to_mp3_cmd.split(' '))
    except FileNotFoundError as e:
        print(e)
        print("Check again whether you have fluidsynth and lame installed in "
              "your system.")
        sys.exit(1)
    subprocess.call(rm_wavfile_cmd.split(' '))



def generate_audio_file(filename, soundfont_path=None):
    policies = load_obj('POLICIES')
    lambdas = load_obj('LAMBDAS')
    start_states = load_obj('START_STATES')
    term_states = load_obj('TERM_STATES')
    fignotes_dict = load_obj('FIGNOTES_DICT')
    chords_dict = load_obj('CHORDS_DICT')
    dir = 'output/'
    output_path = dir+filename

    make_dir_when_not_exist(dir)
    policy = mix_policies(policies, lambdas)
    trajectory = generate_trajectory(choice(list(start_states)), term_states,
                                policy)
    song = get_original_state(trajectory, fignotes_dict, chords_dict)
    score = translate_states_to_song(song)
    score.write('midi', output_path + '.mid')
    print("{}.midi was created and saved to {} directory.".format(filename,
                                                                 dir))
    if soundfont_path is None:
        score.write('musicxml', output_path+'.xml')
        print("{}.xml was created and saved to {} directory.".format(filename,
                                                                 dir))
    else:
        convert_midi_to_mp3_file(output_path, soundfont_path)
        print("{}.mp3 was created and saved to {} directory.".format(filename,
                                                                 dir))


if __name__ == '__main__':
    generate_audio_file('out', 'TimGM6mb.sf2')
