# -*- coding: utf-8 -*-
import glob
import os
import sys

import keras

# adding current dir to lib path
mydir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, mydir)

from utility.save import check_dir

class Model():
    def __init__(self, application, input_shape, class_num, epochs, batch_size, lr, verbose=False):
        self.application = application
        self.input_shape = input_shape
        self.class_num = class_num
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose

        self.__create_model()
        self.__set_optimizer()

    def __create_model(self):
        """create model
        
        Args:
            class_num (int): output class number
            verbose (bool): output debug information
        
        Returns:
            model: keras model object
        """
        if self.verbose:
            print('input image size: {}'.format(self.input_shape))
        from model.main import generate_model_base
        model = generate_model_base(
            preset=self.application,
            width=self.input_shape[0],
            height=self.input_shape[1],
            channel=self.input_shape[2],
            class_num=self.class_num,
            weights_init=None
        )

        if self.verbose:
            model.summary()

        self.model = model

    
    def __set_optimizer(self):
        from keras.optimizers import Adagrad
        optimizer = Adagrad(lr=self.lr.init)

        self.model.compile(
            loss='categorical_crossentropy', 
            optimizer=optimizer, 
            metrics=['accuracy']
            )

    
    def train(self, sample, out_dir):
        """training
        
        Args:
            sample (Sample): image and label data
        """
        # set callback
        callbacks = self.__callbacks(out_dir)
        
        class_train, class_val = sample.get_one_hot()

        # self.model.fit(
        #     sample.image_train,
        #     class_train,
        #     self.batch_size,
        # )
        from model_action import train as model_train

        model_train(
            self.model,
            sample.image_train,
            sample.image_val,
            class_train,
            class_val,
            self.epochs,
            self.batch_size,
            callbacks=callbacks
        )

    def predict(self, image_unlabeled, out_dir):

        unlabeled_count = image_unlabeled.shape[0] if image_unlabeled is not None else 0
        result_path = os.path.join(out_dir, 'predict_{:010d}.csv'.format(unlabeled_count))

        from model_action import predict as model_predict

        result = model_predict(
            self.model,
            image_unlabeled,
            result_path,
            batch_size=self.batch_size,
            verbose=self.verbose
        )
        return result


    def save(self, out_dir):
        save_model_path = os.path.join(out_dir, 'trained.json')
        self.save_model(save_model_path)
        save_weight_path = os.path.join(out_dir, 'trained.h5')
        self.save_weight(save_weight_path)


    def __callbacks(self, out_dir):
        callbacks = []

        # history
        from callback.history import History
        history_save_path = os.path.join(out_dir, 'history.csv')
        history_callback = History(history_save_path)
        callbacks.append(history_callback)

        # learning rate schedule
        if self.lr is not None:
            from callback.lr_schedule import LearningRateScheduler
            lr_schedule_callback = LearningRateScheduler(
                initial_rate=self.lr.init,
                step_rate=self.lr.step,
                step_epoch=self.lr.epoch
            )
            callbacks.append(lr_schedule_callback)

        # verbose
        if self.verbose:
            from callback.verbose import Verbose
            verbose_callback = Verbose()
            callbacks.append(verbose_callback)

        return callbacks

    def save_model(self, path):
        check_dir(path)
        json_string = self.model.to_json()
        with open(path, 'w') as f:
            f.write(json_string)

    def save_weight(self, path):
        check_dir(path)
        self.model.save_weights(path)


