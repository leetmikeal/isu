# -*- coding: utf-8 -*-
import glob
import os
import sys

import cv2
import keras
import numpy as np
from tqdm import tqdm


def setup_argument_parser(parser):
    """
    Set argument
    """
    # parser.add_argument(
    #     '--in-dir',
    #     help='destination directory path',
    #     required=True)
    # parser.add_argument(
    #     '--in-weight',
    #     help='initial model weight')
    # parser.add_argument(
    #     '--cache-image',
    #     help='cache image saving directory path')
    # parser.add_argument(
    #     '--out-dir',
    #     help='output files save directory path',
    #     required=True)
    # parser.add_argument(
    #     '--epochs',
    #     help='epoch count',
    #     type=int,
    #     required=True)
    # parser.add_argument(
    #     '--batch-size',
    #     help='batch size',
    #     type=int,
    #     default=64)
    # parser.add_argument(
    #     '--application',
    #     help='deep learning structure [isensee2017, unet]',
    #     default='isensee2017')
    # parser.add_argument(
    #     '--sample-init',
    #     help='initial number of image sample',
    #     type=int,
    #     default=400)
    # parser.add_argument(
    #     '--sample-val',
    #     help='number of validation image sample',
    #     type=int,
    #     default=100)
    # parser.add_argument(
    #     '--sample-crop',
    #     help='crop size',
    #     type=int,
    #     default=64)
    # parser.add_argument(
    #     '--lr-init',
    #     help='initial learning rate',
    #     type=float,
    #     default=0.01)
    # parser.add_argument(
    #     '--lr-step',
    #     help='learning rate chagning value',
    #     type=float,
    #     default=0.1)
    # parser.add_argument(
    #     '--lr-epochs',
    #     help='learning rate keep epoch',
    #     type=int,
    #     default=30)
    # parser.add_argument(
    #     '--verbose',
    #     help='output process detail',
    #     action='store_true')



def training():
    """training and validation

    Args:
    """
    raise ValueError('not implemented training 2d')
    print('completed!')


def main(args):
    training(
    )


if __name__ == '__main__':
    test_args = type("Hoge", (object,), {
        # 'save_dir': 'work/mnist',
        # 'verbose': True,
    })
    main(test_args)
