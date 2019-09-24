import os
import glob
import numpy as np
import keras
import matplotlib.pyplot as plt

from model import UNet
import dataset
import tensorflow as tf

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.8))
session = tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(session)


def main():
    unet = UNet(inifile='isu/2d/setting.ini')   
    ds = dataset.Dataset()
    
    width, height, imgnum = ds.load_csv(unet.csv_path)
    X_train = ds.image_read(unet.input_path+'/*.tif', width, height)
    mwidth, mheight = X_train.shape[1], X_train.shape[2]
    X_train_norm = X_train.astype(np.float32)/255.
    X_train_norm = np.pad(X_train_norm, [(0,0),(0,unet.max_size-mwidth),(0,unet.max_size-mheight)], 'constant')
    X_train_norm = X_train_norm.reshape(-1, unet.max_size, unet.max_size, 1)
    print('X_train_norm:', X_train_norm.shape)
    
    model = keras.models.load_model(unet.model_name, custom_objects={'loss': unet.soft_dice_loss()})
    print(model.summary())
    
    Y_pred = model.predict(X_train_norm, batch_size=unet.predict_batch_size, verbose=0)
    Y_pred_bin = np.copy(Y_pred)
    Y_pred_bin[Y_pred_bin<0.5] = 0
    Y_pred_bin[Y_pred_bin>=0.5] = 1
    Y_pred_bin = (Y_pred_bin*255).reshape(-1, unet.max_size, unet.max_size).astype(np.uint8)
    Y_pred_bin = Y_pred_bin[:, :mwidth, :mheight]
    output = np.rollaxis(Y_pred_bin, axis=0, start=3)

    if not os.path.exists(unet.temp2d_path):
        if not os.path.exists(unet.temp_dir):
            os.mkdir(unet.temp_dir)
        os.mkdir(unet.temp2d_path)
    if not os.path.exists(unet.pre_path):
        os.mkdir(unet.pre_path)
    outPath = unet.pre_path + '/' + unet.dataset + '_temp_'
    ds.image_save(output, outPath)
    
if __name__ == '__main__':
    main()
    
