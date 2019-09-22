# -*- coding: utf-8 -*-
import numpy as np
from keras.callbacks import Callback


class Verbose(Callback):
    def __init__(self, n_image_train, n_image_val):
        self.n_image_train = n_image_train
        self.n_image_val = n_image_val

    def on_epoch_end(self, epoch, logs=None):
        # print('train {:03d} (loss:{:.5f}, acc:{:.5f}), val (loss: {:.5f}, acc:{:.5f})'.format(
        #     epoch,
        #     logs['train_loss'],
        #     logs['train_acc'],
        #     logs['val_loss'],
        #     logs['val_acc']
        # ))
        # print('train {:03d} (loss:{:.5f}), val (loss: {:.5f})'.format(
        #     epoch,
        #     logs['train_loss'],
        #     logs['val_loss'],
        # ))
        # print('train {:03d} (loss:{:.5f}), val (loss: {:.5f})'.format(
        #     epoch,
        #     logs['loss'],
        #     logs['val_loss'],
        # ))

        train_batch_returns = logs['batch_train_history']
        val_batch_returns = logs['batch_val_history']
        # print('train : {}'.format(train_batch_returns))
        # print('val : {}'.format(val_batch_returns))

        train_loss = np.sum(np.array(train_batch_returns)[:, 0]) / self.n_image_train
        val_loss = np.sum(np.array(val_batch_returns)[:, 0]) / self.n_image_val
        print('{:>4d} train : {}'.format(epoch, train_loss))
        print('{:>4d} val : {}'.format(epoch, val_loss))

