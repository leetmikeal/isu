# -*- coding: utf-8 -*-
import glob
import os
import sys

import cv2
import keras
import numpy as np
from tqdm import tqdm

from utility.save import check_dir
from core import Model, SampleSingle


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
        default=1)
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

    # load images
    sample = SampleSingle(
        image_dir_path=sample_dir,
        verbose=verbose)

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
        input_shape=sample.input_shape(),
        verbose=verbose
    )

    # # debug output
    # if verbose:
    #     model.save_plot_model(os.path.join(out_dir, 'model.png'))

    # training
    input_sample = np.array([sample.image])
    result = model.predict(input_sample, out_dir)

    # # save model
    # result.save(out_dir)
    save_slice_result(result, sample.padding_position, sample.originl_input_shape(), out_dir)

    print('completed!')


def save_slice_result(nparray, padding_position, input_shape, dir_path):
    print('result saving...')
    print('shape : {}'.format(nparray.shape))

    base_path = os.path.join(dir_path, 'result_images')
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    nparray = nparray[
        0,
        0,
        padding_position[0]:padding_position[0]+input_shape[0],
        padding_position[1]:padding_position[1]+input_shape[1],
        padding_position[2]:padding_position[2]+input_shape[2],
        0
    ]

    for iz in tqdm(range(nparray.shape[2])):
        img = nparray[:, :, iz]
        saved_img = np.zeros(img.shape, dtype=np.uint8)
        saved_img[img > 0.5] = 255

        saving_path = os.path.join(base_path, '{:04d}.tif'.format(iz))
        cv2.imwrite(saving_path, saved_img)


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
