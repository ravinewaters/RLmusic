__author__ = 'redhat'

from preprocess import preprocess
from algorithm import run_AL_algorithm
from music_generation import generate_score
import argparse





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Melody Generator.')
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
    args = parser.parse_args()
    # print(args)

    print('Start preprocessing...')
    preprocess(args.dir)
    print('Done preprocessing...')
    print('Running AL algorithm...')
    run_AL_algorithm(args.disc_rate, args.eps)
    print('Generate Score')
    generate_score()
