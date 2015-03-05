__author__ = 'redhat'

from features_generation import FeaturesPreprocessor
from algorithm import ALAlgorithm
from music_generation import MusicGenerator
from sys import exit
import configparser


def main():
    """
    Melody Generator.

    Dependencies:
    Python packages: music21, numpy, scipy, cvxopt.
    External programs: fluidsynth, lame.
    Requires files: soundfont for fluidsynth
    """

    config = configparser.ConfigParser()
    config.read('config.ini')
    try:
        corpus_dir = config['preprocessor']['corpus_dir']
        disc_rate = float(config['al_algorithm']['disc_rate'])
        eps = float(config['al_algorithm']['eps'])
        max_reward = float(config['al_algorithm']['max_reward'])
        outfile = config['music_generator']['output_filename']
        outdir = config['music_generator']['output_directory']
        format = config['music_generator']['output_format']
        soundfont_path = config['music_generator']['soundfont_path']
    except KeyError as e:
        print(e.args, 'parameter is not found in config.ini')
        exit(1)

    preprocessor = FeaturesPreprocessor()
    preprocessor.run(corpus_dir)
    al_algorithm = ALAlgorithm(preprocessor)
    al_algorithm.run(disc_rate, eps, max_reward)
    music_generator = MusicGenerator(preprocessor,
                                     al_algorithm)
    music_generator.run(outfile, outdir, soundfont_path, format)

if __name__ == '__main__':
    main()