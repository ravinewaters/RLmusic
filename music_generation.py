__author__ = 'redhat'

from common_methods import compute_next_state, load_obj, weighted_choice_b, \
    make_dir_when_not_exist
from random import choice
import subprocess
import music21
import sys



class MusicGenerator():
    def __init__(self,
                 outfile,
                 soundfont_path=None,
                 format='midi',
                 preprocessor=None,
                 al_algorithm=None):

        out_dir = 'output/'
        make_dir_when_not_exist(out_dir)

        self.out_path = out_dir + outfile
        self.soundfont_path = soundfont_path
        self.format = format

        # need fix
        if preprocessor is None or al_algorithm is None:
            self.policies = load_obj('POLICIES')
            self.lambdas = load_obj('LAMBDAS')
            self.start_states = load_obj('START_STATES')
            self.term_states = load_obj('TERM_STATES')
            self.fignotes_dict = load_obj('FIGNOTES_DICT')
            self.chords_dict = load_obj('CHORDS_DICT')
        else:
            self.policies = al_algorithm.policies
            self.lambdas = al_algorithm.lambdas
            self.start_states = preprocessor.start_states
            self.term_states = preprocessor.term_states
            self.fignotes_dict = preprocessor.fignotes_dict
            self.chords_dict = preprocessor.chords_dict

    def mix_policies(self):
        idx = weighted_choice_b(self.lambdas)
        return self.policies[idx]


    def generate_trajectory(self):
        """
        Generate trajectory using start_state as the start state and
        policy_matrix as the policy until landing on term_states and action == -1.
        """
        state = choice(list(self.start_states))
        original_trajectory = []
        counter = 0
        while counter <= 30:
            action = choice(self.policy[state])
            if state in self.term_states and action == -1:
                break
            state = compute_next_state(state, action)
            original_state = (state[0],
                              self.fignotes_dict[1][state[1]],
                              self.chords_dict[1][state[2]],
                              int(state[3] / 4.0),
                              int(state[4] / 4.0),
                              state[5])
            original_trajectory.append(original_state)
            counter += 1
        return original_trajectory

    def translate_trajectory_to_music21(self, title='', composer=''):

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

        for state in self.trajectory:
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


    def convert_midi_to_mp3_file(self):
        # requires fluidsynth and lame

        wavfile = self.out_path+'.wav'
        midifile = self.out_path+'.mid'
        midi_to_wav_cmd = "fluidsynth -F {} {} {}".format(wavfile,
                                                          self.soundfont_path,
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

    def write(self):
        avail_format = ('midi', 'musicxml', 'mp3')

        # write midi
        if self.format == 'midi' or format is None:
            filename = self.out_path + '.mid'
            self.score.write('midi', filename)
            print("File saved to {}.".format(filename))

        elif self.format == 'musicxml':
            filename = self.out_path + '.xml'
            self.score.write('musicxml', filename)
            print("File saved to {}.".format(filename))

        elif self.format == 'mp3':
            if self.soundfont_path is None:
                print("You do not have soundfont configured.")
            else:
                filename = self.out_path + '.mp3'
                self.convert_midi_to_mp3_file()
                print("File saved to {}.".format(filename))
        else:
            print("Wrong format chosen. Available formats: {}.".format(
                avail_format))

    def run(self):
        # randomly chosen policy
        self.policy = self.mix_policies()
        self.trajectory = self.generate_trajectory()
        self.score = self.translate_trajectory_to_music21()
        self.write()

if __name__ == '__main__':
    # generate_audio_file('out', 'TimGM6mb.sf2')
    m = MusicGenerator('out', 'TimGM6mb.sf2', 'musicxml')
    m.run()