# -*- coding: utf-8 -*-
from keras.callbacks import Callback


class Verbose(Callback):
    def __init__(self):
        pass

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
        print('train {:03d} (loss:{:.5f}), val (loss: {:.5f})'.format(
            epoch,
            logs['loss'],
            logs['val_loss'],
        ))
