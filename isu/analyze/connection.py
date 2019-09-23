import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from utility.save import check_dir

def setup_argument_parser(parser):
    """
    Set argument
    """
    parser.add_argument('--in-dir', help='path', required=True)
    parser.add_argument('--verbose', help='output process detail', action='store_true')



def analyze_connection(in_dir, verbose):

    if verbose:
        print('completed')


def main(args):
    analyze_connection(
        in_dir=args.in_dir,
        verbose=args.verbose
    )


if __name__ == '__main__':
    test_args = type("Hoge", (object,), {
        'save_dir': 'work/mnist',
        'verbose': True,
    })
    main(test_args)
