# -*- coding: utf-8 -*-
import glob
import os
import sys
import math

import keras
import numpy as np

# adding current dir to lib path
mydir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, mydir)

from model.unet3d_metrics import (dice_coefficient, dice_coefficient_loss,
                                  get_label_dice_coefficient_function,
                                  weighted_dice_coefficient_loss)
from utility.save import check_dir


class Model3d():
    def __init__(
            self,
            application,
            input_shape,
            epochs,
            batch_size,
            lr,
            initial_weight=None,
            verbose=False):
        self.application = application
        self.input_shape = input_shape
        self.initial_weight = initial_weight
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose

        if (application is None and input_shape is None and epochs is None and batch_size is None and initial_weight is None):
            return

        self.__create_model(self.application, self.input_shape)
        self.__load_weight(self.initial_weight)

        self.__set_optimizer()

    def __create_model(self, application, input_shape):
        """create model

        Args:
            verbose (bool): output debug information

        Returns:
            model: keras model object
        """
        if self.verbose:
            print('input image size: {}'.format(input_shape))

        if application == 'isensee2017':
            from model.isensee2017 import isensee2017_model

            model = isensee2017_model(
                input_shape,
                depth=4,
                n_base_filters=8,
                n_labels=1,
            )

        elif application == 'unet':
            from model.unet import unet_model_3d
            # model = unet_model_3d(
            #     self.input_shape,
            #     depth=3,
            #     n_base_filters=8,
            #     initial_learning_rate=self.lr.init)
            model = unet_model_3d(
                input_shape,
                depth=2,
                n_base_filters=8)

        else:
            raise ValueError(
                'unknwon model name : {}'.format(
                    application))

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
            label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(n_labels)]
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

    def __load_weight(self, initial_weight):
        """load weight from file

        Args:
            initial_weight (string): initial weight path
        """
        if not os.path.exists(initial_weight):
            print('WARNING: initial weight file was not found. | {}'.format(initial_weight))

        self.model.load_weights(initial_weight)
        if self.verbose:
            print('loaded initial weight | {}'.format(initial_weight))

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
        from model.model_action import train as model_train

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
                path = os.path.join(save_dir, 'image_{:05d}_{:04d}.png'.format(i, j))
                cv2.imwrite(image, path)

    def predict(self, image_unlabeled, out_dir):

        from model.model_action import predict as model_predict

        result = model_predict(
            self.model,
            image_unlabeled,
            result_dir_path=out_dir,
            batch_size=self.batch_size,
            verbose=self.verbose
        )
        return result

    def predict_crop(self, box, out_dir, crop_size=128, overlap=16):
        shift_size = crop_size - overlap

        box_ranges = []
        for iz in range(int(math.ceil(box.shape[2] / shift_size))):
            for iy in range(int(math.ceil(box.shape[1] / shift_size))):
                for ix in range(int(math.ceil(box.shape[0] / shift_size))):
                    x_start = ix * shift_size
                    x_end = min(ix * shift_size + crop_size, box.shape[0])
                    y_start = iy * shift_size
                    y_end = min(iy * shift_size + crop_size, box.shape[1])
                    z_start = iz * shift_size
                    z_end = min(iz * shift_size + crop_size, box.shape[2])
                    if x_end - x_start <= overlap or y_end - y_start <= overlap or z_end - z_start <= overlap:
                        continue
                    box_ranges.append([x_start, x_end, y_start, y_end, z_start, z_end])

        box_parts = []
        for x1, x2, y1, y2, z1, z2 in box_ranges:
            part = box[x1:x2, y1:y2, z1:z2, :]
            x_pad = (0, crop_size - (x2 - x1))
            y_pad = (0, crop_size - (y2 - y1))
            z_pad = (0, crop_size - (z2 - z1))
            part = np.pad(part, (x_pad, y_pad, z_pad, (0, 0)), mode='edge')
            box_parts.append(part)
        box_parts = np.array(box_parts)


        from model.model_action import predict as model_predict

        result_parts = model_predict(
            self.model,
            box_parts,
            result_dir_path=out_dir,
            batch_size=self.batch_size,
            verbose=self.verbose
        )

        result_box = np.zeros(box.shape, dtype=np.uint8)
        half = int(overlap / 2)
        for part, [x1, x2, y1, y2, z1, z2] in zip(result_parts, box_ranges):
            bx_start = x1 if x1 == 0            else x1 + half
            bx_end   = x2 if x2 == box.shape[0] else x2 - half
            by_start = y1 if y1 == 0            else y1 + half
            by_end   = y2 if y2 == box.shape[1] else y2 - half
            bz_start = z1 if z1 == 0            else z1 + half
            bz_end   = z2 if z2 == box.shape[2] else z2 - half
            px_start = 0       if x1 == 0            else half
            px_end   = x2 - x1 if x2 == box.shape[0] else x2 - x1 - half
            py_start = 0       if y1 == 0            else half
            py_end   = y2 - y1 if y2 == box.shape[1] else y2 - y1 - half
            pz_start = 0       if z1 == 0            else half
            pz_end   = z2 - z1 if z2 == box.shape[2] else z2 - z1 - half
            result_box[bx_start:bx_end, by_start:by_end, bz_start:bz_end, :]  \
                = part[px_start:px_end, py_start:py_end, pz_start:pz_end, :]

        return result_box






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

    @staticmethod
    def from_file(path, batch_size, input_shape, verbose=False):
        basepath = os.path.join(os.path.dirname(path), os.path.splitext(os.path.basename(path))[0])
        # structure_path = basepath + '.json'
        weight_path = basepath + '.h5'
        # if not os.path.exists(structure_path):
        #     raise ValueError('file was not found : {}'.format(structure_path))
        if not os.path.exists(weight_path):
            raise ValueError('file was not found : {}'.format(weight_path))

        from model.isensee2017 import isensee2017_model

        model = isensee2017_model(
            input_shape,
            depth=4,
            n_base_filters=8,
            n_labels=1,
        )
        # self.__create_model('isensee2017', input_shape)
        # from keras.models import model_from_json
        # model = model_from_json(structure_path)
        model.load_weights(weight_path)

        new_model = Model3d(None, None, None, None, None)
        new_model.model = model
        new_model.batch_size = batch_size
        new_model.verbose = verbose
        return new_model
