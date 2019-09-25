# -*- coding: utf-8 -*-
import glob
import os
import sys

import cv2
import keras
import numpy as np
from tqdm import tqdm

from core import Config, Model3d, SampleSingle


def setup_argument_parser(parser):
    """
    Set argument
    """
    parser.add_argument(
        '--in-settings',
        help='set setting file path [settings.ini]',
        required=True)
    parser.add_argument(
        '--dataset',
        help='overwrite dataset name in settings.ini by command',
        default=None)
    parser.add_argument(
        '--application',
        help='deep learning structure [isensee2017, unet]',
        default='isensee2017')
    parser.add_argument(
        '--verbose',
        help='output process detail',
        action='store_true')


def predict(
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

    import tensorflow as tf
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.8))
    session = tf.Session(config=config)
    keras.backend.tensorflow_backend.set_session(session)

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

    os.makedirs(out_dir, exist_ok=True)

    # # learning rate
    # lr = LearningRateSchedulerConf(
    #     init=lr_init,
    #     epoch=lr_epochs,
    #     step=lr_step
    # )

    # create model
    model = Model3d.from_file(
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
    os.makedirs(base_path, exist_ok=True)

    nparray = nparray[
        0,
        0,
        padding_position[0]:padding_position[0] + input_shape[0],
        padding_position[1]:padding_position[1] + input_shape[1],
        padding_position[2]:padding_position[2] + input_shape[2],
        0
    ]

    for iz in tqdm(range(nparray.shape[2])):
        img = nparray[:, :, iz]
        saved_img = np.zeros(img.shape, dtype=np.uint8)
        saved_img[img > 0.5] = 255

        saving_path = os.path.join(base_path, '{:04d}.tif'.format(iz))
        cv2.imwrite(saving_path, saved_img)


def main(args):
    config = Config(args.in_settings)
    config.init(args.dataset)
    sample_dir = config.input_path
    model_path = config.model_3d_path
    out_dir = config.output_path
    batch_size = config.predict_3d_batch_size

    predict(
        sample_dir=sample_dir,
        model_path=model_path,
        out_dir=out_dir,
        batch_size=batch_size,
        application=args.application,
        verbose=args.verbose
    )


if __name__ == '__main__':
    test_args = type("Hoge", (object,), {
        'save_dir': 'work/mnist',
        'verbose': True,
    })
    main(test_args)
