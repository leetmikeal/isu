# -*- coding: utf-8 -*-
import glob
import os
import sys

import cv2
import keras
import numpy as np
from tqdm import tqdm

from utility.save import check_dir
from core import Model3d, Sample


def setup_argument_parser(parser):
    """
    Set argument
    """
    parser.add_argument(
        '--in-dir',
        help='destination directory path',
        required=True)
    parser.add_argument(
        '--in-weight',
        help='initial model weight')
    parser.add_argument(
        '--cache-image',
        help='cache image saving directory path')
    parser.add_argument(
        '--out-dir',
        help='output files save directory path',
        required=True)
    parser.add_argument(
        '--epochs',
        help='epoch count',
        type=int,
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
        '--sample-init',
        help='initial number of image sample',
        type=int,
        default=400)
    parser.add_argument(
        '--sample-val',
        help='number of validation image sample',
        type=int,
        default=100)
    parser.add_argument(
        '--sample-crop',
        help='crop size',
        type=int,
        default=64)
    parser.add_argument(
        '--lr-init',
        help='initial learning rate',
        type=float,
        default=0.01)
    parser.add_argument(
        '--lr-step',
        help='learning rate chagning value',
        type=float,
        default=0.1)
    parser.add_argument(
        '--lr-epochs',
        help='learning rate keep epoch',
        type=int,
        default=30)
    parser.add_argument(
        '--verbose',
        help='output process detail',
        action='store_true')


class LearningRateSchedulerConf():
    def __init__(self, init, epoch, step=0.1):
        self.init = init  # initial rate
        self.epoch = epoch
        self.step = step  # step rate


def training(
        sample_dir,
        initial_weight,
        out_dir,
        cache_image,
        epochs,
        batch_size,
        application,
        sample_init,
        sample_val,
        sample_crop,
        lr_init,
        lr_step,
        lr_epochs,
        verbose=False):
    """training and validation

    Args:
        sample_dir (string): sample directory path
        initial_weight (string): model initial weight path
        out_dir (string): output result saving directory path
        cache_imgae (string): cache image directry path
        epochs (int): number of epoch
        bach_size (int): image batch size per epoch
        application (string): model structure [isensee2017, unet]
        sample_init (int): number of initial training sample
        sample_val (int): number of validation image
        sample_crop (int): crop size
        lr_init (float): learning rate initial value
        lr_step (float): learning rate changing value
        lr_epochs (int): number of epoch to keep learning rate value
        verbose (boolean): output debug information
    """
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # load images
    image_dir_path = os.path.join(sample_dir, 'input')
    label_dir_path = os.path.join(sample_dir, 'label')
    sample = Sample(
        image_dir_path=image_dir_path,
        label_dir_path=label_dir_path,
        cache_image=cache_image,
        #crop_size=(64, 64, 64),
        #crop_size=(128, 128, 128),
        crop_size=(sample_crop,) * 3,
        data_count=sample_init + sample_val,
        verbose=verbose)
    sample.load()

    # split data
    sample.split(
        train_count=sample_init,
        val_count=sample_val,
        val_biased=True
    )

    os.makedirs(out_dir, exist_ok=True)

    # learning rate
    lr = LearningRateSchedulerConf(
        init=lr_init,
        epoch=lr_epochs,
        step=lr_step
    )

    # create model
    model = Model3d(
        application=application,
        input_shape=sample.input_shape(),
        initial_weight=initial_weight,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        verbose=verbose
    )

    # # debug output
    # if verbose:
    #     model.save_plot_model(os.path.join(out_dir, 'model.png'))

    # training
    model.train(sample, out_dir)

    # save model
    model.save(out_dir)

    print('completed!')


def main(args):
    training(
        sample_dir=args.in_dir,
        out_dir=args.out_dir,
        initial_weight=args.in_weight,
        cache_image=args.cache_image,
        epochs=args.epochs,
        batch_size=args.batch_size,
        application=args.application,
        sample_init=args.sample_init,
        sample_val=args.sample_val,
        sample_crop=args.sample_crop,
        lr_init=args.lr_init,
        lr_step=args.lr_step,
        lr_epochs=args.lr_epochs,
        verbose=args.verbose
    )


if __name__ == '__main__':
    test_args = type("Hoge", (object,), {
        'save_dir': 'work/mnist',
        'verbose': True,
    })
    main(test_args)
