import glob
import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from core import Dataset2d
from core import Config
from core import Model2d

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.8))
session = tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(session)


def setup_argument_parser(parser):
    """
    Set argument
    """
    parser.add_argument(
        '--in-settings',
        help='set setting file path [settings.ini]',
        required=True)
    parser.add_argument(
        '--verbose',
        help='output process detail',
        action='store_true')


def predict(in_settings, verbose=False):
    config = Config(in_settings)
    unet = Model2d(settings)   
    ds = Dataset2d()
    
    width, height, imgnum = ds.load_csv(config.csv_path)
    X_train = ds.image_read(config.input_path+'/*.tif', width, height)
    mwidth, mheight = X_train.shape[1], X_train.shape[2]
    X_train_norm = X_train.astype(np.float32)/255.
    X_train_norm = np.pad(X_train_norm, [(0,0),(0,config.max_size-mwidth),(0,config.max_size-mheight)], 'constant')
    X_train_norm = X_train_norm.reshape(-1, config.max_size, config.max_size, 1)
    print('X_train_norm:', X_train_norm.shape)
    
    model = keras.models.load_model(config.model_name, custom_objects={'loss': config.soft_dice_loss()})
    print(model.summary())
    
    Y_pred = model.predict(X_train_norm, batch_size=config.predict_batch_size, verbose=0)
    Y_pred_bin = np.copy(Y_pred)
    Y_pred_bin[Y_pred_bin<0.5] = 0
    Y_pred_bin[Y_pred_bin>=0.5] = 1
    Y_pred_bin = (Y_pred_bin*255).reshape(-1, config.max_size, config.max_size).astype(np.uint8)
    Y_pred_bin = Y_pred_bin[:, :mwidth, :mheight]
    output = np.rollaxis(Y_pred_bin, axis=0, start=3)

    if not os.path.exists(config.temp2d_path):
        if not os.path.exists(config.temp_dir):
            os.mkdir(config.temp_dir)
        os.mkdir(config.temp2d_path)
    if not os.path.exists(config.pre_path):
        os.mkdir(config.pre_path)
    outPath = config.pre_path + '/' + config.dataset + '_temp_'
    ds.image_save(output, outPath)
    
def main(args):
    predict(
        in_settings=args.in_settings,
        verbose=args.verbose
    )


if __name__ == '__main__':
    test_args = type("Hoge", (object,), {
        'save_dir': 'work/mnist',
        'verbose': True,
    })
    main(test_args)
