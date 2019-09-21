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
    parser.add_argument('--result-dir', help='prefix of result directory path', required=True)
    parser.add_argument('--out-path', help='output csv path', required=True)
    parser.add_argument('--sample-init', help='initial number of sample image', type=int, required=True)
    parser.add_argument('--sample-step', help='step number of adding sample image', type=int, required=True)
    parser.add_argument('--verbose', help='output process detail', action='store_true')


def get_file_list(dirpath):
    for p in glob.iglob(os.path.join(dirpath, '*', 'history.csv')):
        # ex.) work/output_cifar10_01/00/history.csv
        yield p


def get_last_epoch_value(path):
    df = pd.read_csv(path, encoding='utf-8')
    #last_row = df.iloc[-1]
    last_row = df.tail(1)
    return last_row


def analyze_last_precision(result_dir, out_path, sample_init, sample_step, verbose):
    file_list = get_file_list(result_dir)

    last_list = []
    for f in tqdm(file_list):
        best_val = get_last_epoch_value(f)
        last_list.append(best_val)

    sample = pd.Series(
        [sample_init + i * sample_step for i in range(len(last_list))],
        index=list(range(len(last_list))),
        name='sample'
    )

    df = pd.concat(last_list, ignore_index=True)
    df = pd.concat([sample, df], axis=1)
    df = df.drop('epoch', axis=1)

    check_dir(out_path)
    df.to_csv(out_path, encoding='utf-8')

    if verbose:
        print('completed')


def main(args):
    analyze_last_precision(
        result_dir=args.result_dir,
        out_path=args.out_path,
        sample_init=args.sample_init,
        sample_step=args.sample_step,
        verbose=args.verbose
    )


if __name__ == '__main__':
    test_args = type("Hoge", (object,), {
        'save_dir': 'work/mnist',
        'verbose': True,
    })
    main(test_args)
