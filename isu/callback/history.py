# -*- coding: utf-8 -*-
import numpy as np
from keras.callbacks import Callback

from utility.save import check_dir


class History(Callback):
    def __init__(self, save_path, n_image_train, n_image_val):
        self.save_path = save_path
        self.n_image_train = n_image_train
        self.n_image_val = n_image_val
        self.epochs = []
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        train_batch_returns = logs['batch_train_history']
        val_batch_returns = logs['batch_val_history']
        train_loss = np.sum(np.array(train_batch_returns)[:, 0]) / self.n_image_train
        val_loss = np.sum(np.array(val_batch_returns)[:, 0]) / self.n_image_val

        self.epochs.append(epoch)
        self.history.append({'train_loss':train_loss, 'val_loss':val_loss})

    def on_train_end(self, logs=None):
        import pandas as pd
        epoch_logs_list = self.history
        train_loss = [epoch_logs['train_loss'] for epoch_logs in epoch_logs_list]
        # train_acc = [epoch_logs['train_acc'] for epoch_logs in epoch_logs_list]
        val_loss = [epoch_logs['val_loss'] for epoch_logs in epoch_logs_list]
        # val_acc = [epoch_logs['val_acc'] for epoch_logs in epoch_logs_list]

        sr_epoch_index = pd.Series(self.epochs)
        sr_train_loss = pd.Series(train_loss)
        # sr_train_acc = pd.Series(train_acc)
        sr_val_loss = pd.Series(val_loss)
        # sr_val_acc = pd.Series(val_acc)

        sr_epoch_index.name = 'epoch'
        sr_train_loss.name = 'train_loss'
        # sr_train_acc.name = 'train_acc'
        sr_val_loss.name = 'val_loss'
        # sr_val_acc.name = 'val_acc'

        #df_histoty = pd.concat([sr_epoch_index, sr_train_loss, sr_train_acc, sr_val_loss, sr_val_acc], axis=1)
        df_histoty = pd.concat([sr_epoch_index, sr_train_loss, sr_val_loss], axis=1)
        check_dir(self.save_path)
        df_histoty.to_csv(self.save_path, index=False)