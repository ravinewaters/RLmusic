__author__ = 'redhat'

from features_generation import FeaturesPreprocessor
from algorithm import ALAlgorithm
from music_generation import MusicGenerator
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
    corpus_dir = config['preprocessor']['corpus_dir']
    disc_rate = float(config['al_algorithm']['disc_rate'])
    eps = float(config['al_algorithm']['eps'])
    output_filename = config['music_generator']['output_filename']
    soundfont_path = config['music_generator']['soundfont_path']
    output_format = config['music_generator']['format']

    preprocessor = FeaturesPreprocessor(corpus_dir)
    preprocessor.run()
    al_algorithm = ALAlgorithm(disc_rate, eps, preprocessor)
    al_algorithm.run()
    music_generator = MusicGenerator(output_filename,
                                     soundfont_path,
                                     output_format,
                                     preprocessor,
                                     al_algorithm)
    music_generator.run()


if __name__ == '__main__':
    main()