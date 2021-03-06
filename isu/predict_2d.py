import glob
import math
import os

import keras
import matplotlib.pyplot as plt
import numpy as np

from core import Config, Dataset2d, Model2d


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
        '--verbose',
        help='output process detail',
        action='store_true')


def predict(in_settings, overwrite_dataset=None, verbose=False):
    import tensorflow as tf
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.8))
    session = tf.Session(config=config)
    keras.backend.tensorflow_backend.set_session(session)

    # config
    config = Config(in_settings)
    config.init(overwrite_dataset)
    if verbose:
        config.debug()

    # loading sample
    ds = Dataset2d()
    width, height, imgnum = ds.load_csv(os.path.join(config.input_path, 'input.csv'))
    model_mn = 16  # model magick number
    width_ex = int(math.ceil(width / model_mn) * model_mn)
    height_ex = int(math.ceil(height / model_mn) * model_mn)
    x_train = ds.image_read(os.path.join(config.input_path, '*.tif'), width, height)
    x_train_norm, x_width, x_height = reshape(x_train, (width_ex, height_ex))

    # create model
    model_base = Model2d(config)
    model = model_base.load(input_shape=(width_ex, height_ex, 1))

    if verbose:
        print('x_train_norm:', x_train_norm.shape)
        print(model.summary())

    # predict
    y_pred = model.predict(x_train_norm, batch_size=config.predict_2d_batch_size, verbose=1 if verbose else 0)
    output = postprocess(y_pred, (width_ex, height_ex), (x_width, x_height))

    # saving
    os.makedirs(config.temp_2d_dir, exist_ok=True)
    ds.image_save(output, config.temp_2d_dir)


def reshape(x_train, size_ex):
    width, height = x_train.shape[1], x_train.shape[2]
    x_train_norm = x_train.astype(np.float32) / 255.
    x_train_norm = np.pad(x_train_norm, [(0, 0), (0, size_ex[0] - width), (0, size_ex[1] - height)], 'constant')
    x_train_norm = x_train_norm.reshape(-1, size_ex[0], size_ex[1], 1)
    return x_train_norm, width, height


def postprocess(y_pred, size_ex, size):
    y_pred_bin = np.copy(y_pred)
    y_pred_bin[y_pred_bin < 0.5] = 0
    y_pred_bin[y_pred_bin >= 0.5] = 1
    y_pred_bin = (y_pred_bin * 255).reshape(-1, size_ex[0], size_ex[1]).astype(np.uint8)
    y_pred_bin = y_pred_bin[:, :size[0], :size[1]]
    output = np.rollaxis(y_pred_bin, axis=0, start=3)
    return output


def main(args):
    predict(
        in_settings=args.in_settings,
        overwrite_dataset=args.dataset,
        verbose=args.verbose
    )


if __name__ == '__main__':
    test_args = type("Hoge", (object,), {
        'save_dir': 'work/mnist',
        'verbose': True,
    })
    main(test_args)
