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
        help='destination directory path',
        required=True)
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
        '--sample-add',
        help='sample addition method [random, confident, unconfident, high_entropy]',
        default='random')
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
    out_dir, 
    cache_image, 
    epochs, 
    batch_size, 
    sample_init, 
    sample_val, 
    lr_init,
    lr_step,
    lr_epochs,
    verbose=False):
    """training and validation

    Args:
        sample_dir (string): sample directory path
        out_dir (string): output result saving directory path
        cache_imgae (string): cache image directry path
        epochs (int): number of epoch
        bach_size (int): image batch size per epoch
        sample_init (int): number of initial training sample
        sample_val (int): number of validation image
        lr_init (float): learning rate initial value
        lr_step (float): learning rate changing value
        lr_epochs (int): number of epoch to keep learning rate value
        verbose (boolean): output debug information
    """

    # load images
    sample = Sample(
        dir_path=sample_dir, 
        cache_image=cache_image, 
        crop_size=(64, 64, 64),
        data_count=sample_init + sample_val,
        verbose=verbose)
    sample.load()

    # split data
    sample.split(
        train_count=sample_init,
        val_count=sample_val,
        val_biased=True
    )

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # learning rate
    lr = LearningRateSchedulerConf(
        init=lr_init,
        epoch=lr_epochs,
        step=lr_step
    )

    # create model
    model = Model(
        application='bench',
        # application='resnet20',
        # application='resnet50',
        input_shape=sample.input_shape(),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        verbose=verbose
        )

    # training
    model.train(sample, out_dir)

    # save model
    model.save(out_dir)


    print('completed!')


def main(args):
    training(
        sample_dir=args.in_dir,
        out_dir=args.out_dir,
        cache_image=args.cache_image,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sample_init=args.sample_init,
        sample_val=args.sample_val,
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
