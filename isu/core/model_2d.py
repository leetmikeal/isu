import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import skimage.transform as trans
from keras import backend as keras
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import *
from keras.models import *
from keras.optimizers import *

# adding current dir to lib path
mydir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, mydir)


class Model2d():   # UNet
    def __init__(self, config):
        self.config = config


    def load(self, input_shape = (400,400,1)):
        model = self.build(input_shape)
        if self.config.model_2d_path is not None:
            model.load_weights(self.config.model_2d_path)
        return model


    def build(self, input_shape = (400,400,1)):
        inputs = Input(shape=input_shape)
        conv1 = Conv2D(64, 3, padding = 'same', kernel_initializer = self.config.ke_init)(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv1 = Conv2D(64, 3, padding = 'same', kernel_initializer = self.config.ke_init)(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, padding = 'same', kernel_initializer = self.config.ke_init)(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        conv2 = Conv2D(128, 3, padding = 'same', kernel_initializer = self.config.ke_init)(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, padding = 'same', kernel_initializer = self.config.ke_init)(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)
        conv3 = Conv2D(256, 3, padding = 'same', kernel_initializer = self.config.ke_init)(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, padding = 'same', kernel_initializer = self.config.ke_init)(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation('relu')(conv4)
        conv4 = Conv2D(512, 3, padding = 'same', kernel_initializer = self.config.ke_init)(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation('relu')(conv4)
        ###drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(1024, 3, padding = 'same', kernel_initializer = self.config.ke_init)(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation('relu')(conv5)
        conv5 = Conv2D(1024, 3, padding = 'same', kernel_initializer = self.config.ke_init)(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = Activation('relu')(conv5)
        ###drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = self.config.ke_init)(UpSampling2D(size = (2,2))(conv5))
        merge6 = concatenate([conv4,up6], axis = 3)
        conv6 = Conv2D(512, 3, padding = 'same', kernel_initializer = self.config.ke_init)(merge6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Activation('relu')(conv6)
        conv6 = Conv2D(512, 3, padding = 'same', kernel_initializer = self.config.ke_init)(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Activation('relu')(conv6)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = self.config.ke_init)(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(256, 3, padding = 'same', kernel_initializer = self.config.ke_init)(merge7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Activation('relu')(conv7)
        conv7 = Conv2D(256, 3, padding = 'same', kernel_initializer = self.config.ke_init)(conv7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Activation('relu')(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = self.config.ke_init)(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(128, 3, padding = 'same', kernel_initializer = self.config.ke_init)(merge8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Activation('relu')(conv8)
        conv8 = Conv2D(128, 3, padding = 'same', kernel_initializer = self.config.ke_init)(conv8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Activation('relu')(conv8)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = self.config.ke_init)(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(64, 3, padding = 'same', kernel_initializer = self.config.ke_init)(merge9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Activation('relu')(conv9)
        conv9 = Conv2D(64, 3, padding = 'same', kernel_initializer = self.config.ke_init)(conv9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Activation('relu')(conv9)
        conv9 = Conv2D(2, 3, padding = 'same', kernel_initializer = self.config.ke_init)(conv9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Activation('relu')(conv9)

        conv10 = Conv2D(1, 1, padding="valid")(conv9)
        conv10 = Reshape((input_shape[0] * input_shape[1],))(conv10)
        conv10 = Activation("sigmoid")(conv10)
        model = Model(input = inputs, output = conv10)

        return model

    
    def weighted_cross_entropy(self, beta=10.):
        def convert_to_logits(y_pred):
            y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
            return tf.log(y_pred / (1 - y_pred))

        def loss(y_true, y_pred):
            y_pred = convert_to_logits(y_pred)
            loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=beta)
            return tf.reduce_mean(loss)

        return loss

    
    def soft_dice_loss(self):
        def dice_coef(y_true, y_pred, smooth=1):
            intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
            return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

        def loss(y_true, y_pred):
            return 1-dice_coef(y_true, y_pred)

        return loss

    
    def plot_result(self, II, IL, IP, IPB, mwidth, mheight):
        row, col = 5, 4
        start, interval =100, 20
        fig, ax = plt.subplots(nrows=row, ncols=col, sharex=True, sharey=True, figsize=(8, 8))
        ax = ax.flatten()

        for i in range(row):
            ax[i*col].imshow(II[start+i*interval, :, :, :].reshape(mwidth, mheight))
            ax[i*col].set_title(str(start+i*interval)+" input")
            ax[i*col+1].imshow(IL[start+i*interval, :].reshape(mwidth, mheight))
            ax[i*col+1].set_title("label")
            ax[i*col+2].imshow(IP[start+i*interval, :].reshape(mwidth, mheight))
            ax[i*col+2].set_title("output")
            ax[i*col+3].imshow(IPB[start+i*interval, :].reshape(mwidth, mheight))
            ax[i*col+3].set_title("bin")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        plt.show()
        return fig

    
    def evaluate(self, ylabel, ypred):
        acc = np.sum(ylabel==ypred).astype(np.float)/ylabel.shape[0]/ylabel.shape[1]
        tpr = np.count_nonzero(ypred==1)/np.count_nonzero(ylabel==1)
        print("Cluster Volume  predict:", np.count_nonzero(ypred==1)," label:", np.count_nonzero(ylabel==1))
        return acc, tpr
