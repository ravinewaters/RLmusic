__author__ = 'redhat'

from features_generation import FeaturesPreprocessor
from music_generation import MusicGenerator
import argparse



def main():
    description="""
    Melody Generator.

    Dependencies:
    Python packages: music21, numpy, scipy, cvxopt.
    External programs: fluidsynth, lame.
    Requires files: soundfont for fluidsynth
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--dir',
                        help='Specify the directory of corpus. '
                             '(default = %(default)s)',
                        type=str,
                        default='corpus/')
    parser.add_argument('--disc_rate',
                        help='Discount rate for AL algorithm. '
                             '(default = %(default)s)',
                        type=float,
                        default=0.9)
    parser.add_argument('--eps',
                        help='Error threshold for AL algorithm. '
                             '(default = %(default)s)',
                        type=float,
                        default=1)
    parser.add_argument('--output',
                        help='Error threshold for AL algorithm. '
                             '(default = %(default)s)',
                        type=str,
                        default='out')
    parser.add_argument('--soundfont',
                        help="Path to soundfont file. Required to convert to mp3")
    args = parser.parse_args()
    # print(args)

    print('Start preprocessing...')
    preprocess(args.dir)
    print('Done preprocessing...')
    print('Running AL algorithm...')
    run_AL_algorithm(args.disc_rate, args.eps)
    print('Generate audio file')
    generate_audio_file(args.output, args.soundfont)

if __name__ == '__main__':
    main()