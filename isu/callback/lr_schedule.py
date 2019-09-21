# -*- coding: utf-8 -*-
import math

import numpy as np
from keras.callbacks import Callback
from keras import  backend as K


class StepwiseSchedule():
    """学習率のスケジュール 段階的に変更する
    """

    def __init__(self, initial_rate, step_rate, step_epoch):
        self.initial_rate = initial_rate
        self.step_rate = step_rate
        self.step_epoch = step_epoch

    def __call__(self, current_epoch):
        return self.calculate_learning_rate(current_epoch, self.initial_rate, self.step_rate, self.step_epoch)

    def calculate_learning_rate(self, current_epoch, initial_rate, step_rate, step_epoch):
        s = int(math.ceil((current_epoch - step_epoch + 1)/step_epoch))
        return initial_rate * step_rate ** s


class LearningRateScheduler(Callback):
    """Learning rate scheduler.

    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and current learning rate
            and returns a new learning rate as output (float).
        verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, initial_rate, step_rate, step_epoch, verbose=0):
        super(LearningRateScheduler, self).__init__()
        self.schedule = StepwiseSchedule(initial_rate, step_rate, step_epoch)
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(K.get_value(self.model.optimizer.lr))
        try:  # new API
            lr = self.schedule(epoch, lr)
        except TypeError:  # old API for backward compatibility
            lr = self.schedule(epoch)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

