__author__ = 'redhat'

from common_methods import compute_next_state, load_obj, \
    make_dir_when_not_exist
from random import choice, random
import bisect
import subprocess
import music21
import sys
import argparse


class MusicGenerator():
    def __init__(self, preprocessor=None, al_algorithm=None):
        if preprocessor is None or al_algorithm is None:
            self.policies = load_obj('POLICIES')
            self.lambdas = load_obj('LAMBDAS')
            self.start_states = load_obj('START_STATES')
            self.fignotes_dict = load_obj('FIGNOTES_DICT')
            self.chords_dict = load_obj('CHORDS_DICT')
        else:
            self.policies = al_algorithm.policies
            self.lambdas = al_algorithm.lambdas
            self.start_states = preprocessor.start_states
            self.fignotes_dict = preprocessor.fignotes_dict
            self.chords_dict = preprocessor.chords_dict

    def mix_policies(self):
        idx = self.weighted_choice_b(self.lambdas)
        return self.policies[idx]

    @staticmethod
    def weighted_choice_b(weights):
        totals = []
        running_total = 0

        for w in weights:
            running_total += w
            totals.append(running_total)

        rnd = random() * running_total
        return bisect.bisect_right(totals, rnd)

    def generate_trajectory(self):
        """
        Generate trajectory using start_state as the start state and
        policy_matrix as the policy until landing on action == -1.
        """

        state = choice(list(self.start_states))
        original_trajectory = []
        counter = 0
        while counter <= 30:
            original_state = (state[0],
                              self.fignotes_dict[1][state[1]],
                              self.chords_dict[1][state[2]],
                              int(state[3] / 4.0),
                              int(state[4] / 4.0),
                              state[5])
            original_trajectory.append(original_state)
            action = choice(self.policy[state])
            if action == -1:
                break
            state = compute_next_state(state, action)
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

        score.metadata.title = title
        score.metadata.composer = composer

        first_figure = self.trajectory[0]
        if first_figure[2] == 'pickup':
            self.trajectory = self.trajectory[1:]
            duration_str = str(int(first_figure[4]))
            stream.append(music21.meter.TimeSignature(duration_str + '/4'))
            self.translate_figure_to_music21(first_figure, stream)

        stream.append(common_time)
        for state in self.trajectory:
            if state[-1] == -1:
                duration_str = state[1][1]
                r = music21.note.Rest(quarterLength=duration_str)
                stream.append(r)
            else:
                self.translate_figure_to_music21(state, stream)
        stream.makeMeasures(inPlace=True)
        part.append(stream)

        score.append(part)
        return score

    @staticmethod
    def translate_figure_to_music21(figure, stream):
        pitches = figure[1][::2]
        durations = figure[1][1::2]
        for pitch, duration in zip(pitches, durations):
            n = music21.note.Note(pitch, quarterLength=duration)
            stream.append(n)

    @staticmethod
    def convert_midi_to_mp3_file(outpath, soundfont_path):
        # requires fluidsynth and lame

        wavfile = outpath + '.wav'
        midifile = outpath + '.mid'
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

    def write(self, outfile, outdir, soundfont_path, format):
        if outdir[-1] != '/':
            outdir += '/'
        make_dir_when_not_exist(outdir)
        outpath = outdir + outfile
        avail_format = ('midi', 'musicxml', 'mp3')

        # always write midi
        filename = outpath + '.mid'
        self.score.write('midi', filename)

        if format == 'midi':
            print("File saved to {}.".format(filename))

        elif format == 'musicxml':
            filename = outpath + '.xml'
            self.score.write('musicxml', filename)
            print("File saved to {}.".format(filename))

        elif format == 'mp3':
            if soundfont_path is None:
                print("You do not have soundfont configured.")
            else:
                filename = outpath + '.mp3'
                self.convert_midi_to_mp3_file(outpath, soundfont_path)
                print("File saved to {}.".format(filename))
        else:
            print("Wrong format chosen. You chose: {}. Available formats: {"
                  "}.".format(format, avail_format))

    def run(self, outfile, outdir, soundfont_path=None, format='midi'):
        self.policy = self.mix_policies()
        self.trajectory = self.generate_trajectory()
        self.score = self.translate_trajectory_to_music21()
        self.write(outfile, outdir, soundfont_path, format)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Music Generation module",
                            description="Generate a melody in the specified "
                                        "format.")
    parser.add_argument('filename')
    parser.add_argument('-d',
                        '--outdir',
                        default='output/',
                        help="The output will be in this directory.")
    parser.add_argument('--sfpath',
                        default='TimGM6mb.sf2',
                        help="Path of soundfont file.")
    parser.add_argument('-f', '--format',
                        default='midi',
                        choices=('musicxml', 'midi', 'mp3'),
                        help="The format of the output file.")
    args = parser.parse_args()

    mg = MusicGenerator()
    mg.run(args.filename, args.outdir, args.sfpath, args.format)