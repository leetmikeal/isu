# -*- coding: utf-8 -*-
import glob
import os
import sys

import keras

# adding current dir to lib path
mydir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, mydir)

from model.unet3d_metrics import (dice_coefficient, dice_coefficient_loss,
                                  get_label_dice_coefficient_function,
                                  weighted_dice_coefficient_loss)
from utility.save import check_dir


class Model():
    def __init__(
            self,
            application,
            input_shape,
            epochs,
            batch_size,
            lr,
            verbose=False):
        self.application = application
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose

        self.__create_model()
        self.__set_optimizer()

    def __create_model(self):
        """create model

        Args:
            verbose (bool): output debug information

        Returns:
            model: keras model object
        """
        if self.verbose:
            print('input image size: {}'.format(self.input_shape))

        if self.application == 'isensee2017':
            from model.isensee2017 import isensee2017_model

            model = isensee2017_model(
                self.input_shape,
                depth=4,
                n_base_filters=8,
                n_labels=1,
            )

        elif self.application == 'unet':
            from model.unet import unet_model_3d
            # model = unet_model_3d(
            #     self.input_shape,
            #     depth=3,
            #     n_base_filters=8,
            #     initial_learning_rate=self.lr.init)
            model = unet_model_3d(
                self.input_shape,
                depth=2,
                n_base_filters=8)

        else:
            raise ValueError(
                'unknwon model name : {}'.format(
                    self.application))

        # from model.main import generate_model_base
        # model = generate_model_base(
        #     preset=self.application,
        #     width=self.input_shape[0],
        #     height=self.input_shape[1],
        #     channel=self.input_shape[2],
        #     weights_init=None
        # )

        if self.verbose:
            model.summary()

        self.model = model

    def save_plot_model(self, save_path):
        from keras.utils import plot_model
        plot_model(self.model, save_path)

    def __set_optimizer(
            self,
            n_labels=1,
            include_label_wise_dice_coefficients=False,
            metrics=dice_coefficient):
        # from keras.optimizers import Adagrad, SGD
        # # optimizer = Adagrad(lr=self.lr.init)
        # optimizer = SGD(lr=self.lr.init, momentum=0.9, decay=1e-4, nesterov=True)

        # self.model.compile(
        #     loss='categorical_crossentropy',
        #     optimizer=optimizer,
        #     metrics=['accuracy']
        #     )
        # self.model.compile(
        #     loss='binary_crossentropy',
        #     optimizer=optimizer
        #     )

        if not isinstance(metrics, list):
            metrics = [metrics]

        if include_label_wise_dice_coefficients and n_labels > 1:
            label_wise_dice_metrics = [
                get_label_dice_coefficient_function(index) for index in range(n_labels)]
            if metrics:
                metrics = metrics + label_wise_dice_metrics
            else:
                metrics = label_wise_dice_metrics

        optimizer = keras.optimizers.Adam(lr=self.lr.init)
        # self.model.compile(optimizer=optimizer, loss=dice_coefficient_loss, metrics=metrics)
        self.model.compile(
            optimizer=optimizer,
            loss=weighted_dice_coefficient_loss,
            metrics=metrics)
        # model.compile(optimizer=Adam(lr=initial_learning_rate, epsilon=None), loss='binary_crossentropy')

    def train(self, sample, out_dir):
        """training

        Args:
            sample (Sample): image and label data
        """
        # set callback
        callbacks = self.__callbacks(sample, out_dir)

        # class_train, class_val = sample.get_one_hot()

        # self.model.fit(
        #     sample.image_train,
        #     class_train,
        #     self.batch_size,
        # )
        from model_action import train as model_train

        # self.model.fit(
        #     x=sample.image_train,
        #     y=sample.class_train,
        #     epochs=self.epochs,
        #     batch_size=self.batch_size,
        #     shuffle=True,
        #     validation_data=(sample.image_val, sample.class_val),
        #     callbacks=callbacks)

        # print('saving input data')
        # self.save_input(os.path.join(out_dir, 'input_train'), sample.image_train)
        # self.save_input(os.path.join(out_dir, 'input_val'), sample.image_val)
        # self.save_input(os.path.join(out_dir, 'label_train'), sample.class_train)
        # self.save_input(os.path.join(out_dir, 'label_train'), sample.class_val)

        model_train(
            self.model,
            image_training=sample.image_train,
            # image_validation=sample.image_train,
            image_validation=sample.image_val,
            label_training=sample.class_train,
            # label_validation=sample.class_train,
            label_validation=sample.class_val,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            save_weight=True  # for debug
        )

    def save_input(self, save_dir, data):
        from tqdm import tqdm
        import cv2
        for i in tqdm(range(data.shape[0])):
            for j in range(data.shape[1]):
                image = ((data[i, j, :, :, 0] + 0.5) * 255.0).astype(np.uint8)
                path = os.path.join(
                    save_dir,
                    'image_{:05d}_{:04d}.png'.format(
                        i,
                        j))
                cv2.imwrite(image, path)

    def predict(self, image_unlabeled, out_dir):

        unlabeled_count = image_unlabeled.shape[0] if image_unlabeled is not None else 0
        result_path = os.path.join(
            out_dir, 'predict_{:010d}.csv'.format(unlabeled_count))

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

    def __callbacks(self, sample, out_dir):
        callbacks = []

        # history
        from callback.history import History
        history_save_path = os.path.join(out_dir, 'history.csv')
        history_callback = History(
            history_save_path,
            sample.image_train.shape[0],
            sample.image_val.shape[0])
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
            verbose_callback = Verbose(
                sample.image_train.shape[0],
                sample.image_val.shape[0])
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
