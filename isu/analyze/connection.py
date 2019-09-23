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
    parser.add_argument('--in-dir', help='path', required=True)
    parser.add_argument('--out-dir', help='path', required=True)
    parser.add_argument('--verbose', help='output process detail', action='store_true')



def analyze_connection(in_dir, out_dir, verbose):
    # unet = UNet(inifile='setting.ini')
    # ds = dataset.Dataset()
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # imagej
    # cython
    # imglyb
    # pyjnius
    # scyjava
    # jnius
    # conda install -c conda-forge imagej
    
    # fiji_path = os.path.join(unet.fiji_dir)
    #fiji_path = '/Users/tamaki/Research/isu/work/Fiji.app'
    fiji_path = r'C:\Users\tamaki\Downloads\fiji-win64\Fiji.app'
    # ij = imagej.init(fiji_path, headless=False)
    ij = imagej.init(fiji_path)
    print(ij.getVersion())
    ij.batchmode = True
    #ij.ui().showUI()
    # if not os.path.exists(unet.output_path):
    #     os.mkdir(unet.output_path)
    from imagej_reader import Process3DOC

    sample_path = os.path.join(in_dir, '0000.tif')
    savedir = out_dir
    savename = 'aaa'
    Process3DOC(ij, sample_path, savedir, savename)

    if verbose:
        print('completed')


def main(args):
    analyze_connection(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        verbose=args.verbose
    )


if __name__ == '__main__':
    test_args = type("Hoge", (object,), {
        'save_dir': 'work/mnist',
        'verbose': True,
    })
    main(test_args)
