# -*- coding: utf-8 -*-
from keras.callbacks import Callback

from utility.save import check_dir


class History(Callback):
    def __init__(self, save_path):
        self.save_path = save_path
        self.epochs = []
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch)
        self.history.append(logs)

    def on_train_end(self, logs=None):
        import pandas as pd
        epoch_logs_list = self.history
        train_loss = [epoch_logs['train_loss'] for epoch_logs in epoch_logs_list]
        train_acc = [epoch_logs['train_acc'] for epoch_logs in epoch_logs_list]
        val_loss = [epoch_logs['val_loss'] for epoch_logs in epoch_logs_list]
        val_acc = [epoch_logs['val_acc'] for epoch_logs in epoch_logs_list]

        sr_epoch_index = pd.Series(self.epochs)
        sr_train_loss = pd.Series(train_loss)
        sr_train_acc = pd.Series(train_acc)
        sr_val_loss = pd.Series(val_loss)
        sr_val_acc = pd.Series(val_acc)

        sr_epoch_index.name = 'epoch'
        sr_train_loss.name = 'train_loss'
        sr_train_acc.name = 'train_acc'
        sr_val_loss.name = 'val_loss'
        sr_val_acc.name = 'val_acc'

        df_histoty = pd.concat([sr_epoch_index, sr_train_loss, sr_train_acc, sr_val_loss, sr_val_acc], axis=1)
        check_dir(self.save_path)
        df_histoty.to_csv(self.save_path, index=False)