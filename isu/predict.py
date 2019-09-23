# -*- coding: utf-8 -*-
import glob
import os
import sys

import cv2
import keras
import numpy as np
from tqdm import tqdm

from utility.save import check_dir
from core import Model, Sample


def setup_argument_parser(parser):
    """
    Set argument
    """
    parser.add_argument(
        '--in-dir',
        help='source directory path',
        required=True)
    parser.add_argument(
        '--in-model',
        help='sorce model file(*.h5|*.json) path',
        required=True)
    parser.add_argument(
        '--out-dir',
        help='output files save directory path',
        required=True)
    parser.add_argument(
        '--batch-size',
        help='batch size',
        type=int,
        default=64)
    parser.add_argument(
        '--application',
        help='deep learning structure [isensee2017, unet]',
        default='isensee2017')
    parser.add_argument(
        '--verbose',
        help='output process detail',
        action='store_true')


def prediction(
        sample_dir,
        model_path,
        out_dir,
        batch_size,
        application,
        verbose=False):
    """training and validation

    Args:
        sample_dir (string): sample directory path
        out_dir (string): output result saving directory path
        bach_size (int): image batch size per epoch
        application (string): model structure [isensee2017, unet]
        verbose (boolean): output debug information
    """
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # load images
    sample = Sample(
        image_dir_path=sample_dir,
        verbose=verbose)
    sample.load()

    # # split data
    # sample.split(
    #     train_count=sample_init,
    #     val_count=sample_val,
    #     val_biased=True
    # )

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # # learning rate
    # lr = LearningRateSchedulerConf(
    #     init=lr_init,
    #     epoch=lr_epochs,
    #     step=lr_step
    # )

    # create model
    model = Model.from_file(
        path=model_path,
        batch_size=batch_size,
        input_shape=sample.image_raw_list[0].shape,
        verbose=verbose
    )

    # # debug output
    # if verbose:
    #     model.save_plot_model(os.path.join(out_dir, 'model.png'))

    # training
    input_sample = np.array([sample.image_raw_list[0]])
    model.predict(input_sample, out_dir)

    # # save model
    # model.save(out_dir)

    print('completed!')


def main(args):
    prediction(
        sample_dir=args.in_dir,
        model_path=args.in_model,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        application=args.application,
        verbose=args.verbose
    )


if __name__ == '__main__':
    test_args = type("Hoge", (object,), {
        'save_dir': 'work/mnist',
        'verbose': True,
    })
    main(test_args)
