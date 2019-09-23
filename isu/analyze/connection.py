import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import imagej

from utility.save import check_dir

def setup_argument_parser(parser):
    """
    Set argument
    """
    parser.add_argument('--in-dir', help='image contained directory path', required=True)
    parser.add_argument('--out-dir', help='path', required=True)
    parser.add_argument('--in-filename', help='filename', default='0000.tif')
    parser.add_argument('--verbose', help='output process detail', action='store_true')



def analyze_connection(in_dir, out_dir, in_filename='0000.tif', verbose=False):
    # unet = UNet(inifile='setting.ini')
    # ds = dataset.Dataset()
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # for windows
    fiji_path = os.getenv('FIJI_PATH', r'C:\Users\tamaki\Downloads\fiji-win64\Fiji.app')
    if not os.path.exists(fiji_path):
        # for mac
        fiji_path = '/Users/tamaki/Research/isu/work/Fiji.app'

    ij = imagej.init(fiji_path)  # taking long time in first time

    if verbose:
        print('fiji imagej version : {}'.format(ij.getVersion()))

    ij.batchmode = True

    #ij.ui().showUI()
    # if not os.path.exists(unet.output_path):
    #     os.mkdir(unet.output_path)

    from imagej_reader import Process3DOC

    sample_path = os.path.join(in_dir, in_filename)
    savedir = out_dir
    savename = 'aaa'
    Process3DOC(ij, sample_path, savedir, savename)

    if verbose:
        print('completed')


def main(args):
    analyze_connection(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        in_filename=args.in_filename,
        verbose=args.verbose
    )


if __name__ == '__main__':
    test_args = type("Hoge", (object,), {
        'save_dir': 'work/mnist',
        'verbose': True,
    })
    main(test_args)
