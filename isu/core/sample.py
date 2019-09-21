# -*- coding: utf-8 -*-
import glob
import os
import sys

import cv2
import keras
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from tqdm import tqdm

class Sample():
    def __init__(self, dir_path, cache_image, verbose):
        self.dir_path = dir_path
        self.cache_image = cache_image
        self.class_num = 0
        self.width = 0
        self.height = 0
        self.channel = 0
        self.verbose = verbose

        self.image_list = None
        self.class_list = None

    def load(self):
        if self.cache_image is not None:
            image_cache_path, class_cache_path = self.__get_cache_file_path()
            if os.path.exists(image_cache_path) and os.path.exists(
                    class_cache_path):
                image_list = np.load(image_cache_path, allow_pickle=False)
                class_list = np.load(class_cache_path, allow_pickle=False)
                if self.verbose:
                    print('!! cached !!')
                    print('loaded image : {}'.format(image_list.shape))
                    print('loaded class : {}'.format(class_list.shape))

                self.image_list = image_list
                self.class_list = class_list
                self.__set_attributes()
                return

        image_list, class_list = self.__load_image(self.dir_path)

        if self.verbose:
            print('loaded image : {}'.format(image_list.shape))
            print('loaded class : {}'.format(class_list.shape))

        if self.cache_image is not None and os.path.exists(self.cache_image):
            image_cache_path, class_cache_path = self.__get_cache_file_path()
            np.save(image_cache_path, image_list, allow_pickle=False)
            np.save(class_cache_path, class_list, allow_pickle=False)

        self.image_list = image_list
        self.class_list = class_list
        self.__set_attributes()


    def __set_attributes(self):
        self.class_num = self.class_list.max() + 1 # TODO: have to calculate kind of class number
        self.width = self.image_list.shape[1]
        self.height = self.image_list.shape[2]
        self.channel = self.image_list.shape[3]


    def __get_cache_file_path(self):
        image_path = os.path.join(self.cache_image, 'image.npy')
        class_path = os.path.join(self.cache_image, 'class.npy')
        return image_path, class_path


    def __load_image(self, dir_path):
        """load images

        Args:
            dir_path (string): loadimage base directory path

        Returns:
            image_list: image list by numpy array
            label_list: class list by numpy array
        """
        def get_class_name(path):
            dirname = os.path.basename(os.path.dirname(path))
            return dirname

        image_list = []
        label_list = []
        for dataset_path in glob.iglob(os.path.join(dir_path, 'input', '*')):
            boxcell = []
            labelcell = []
            for path in tqdm(
                glob.iglob(
                    os.path.join(
                        dataset_path,
                        '*.tif')),
                disable=(
                    not self.verbose)):

                img = self.__load_single_image_from_file(path)
                boxcell.append(img)

            boxcell = np.concatenate(boxcell, axis=2)
            boxcell = np.array(boxcell, dtype=np.float16) / 255.0

            label_path = os.path.join(dir_path, 'label', os.path.basename(dataset_path))
            for path in tqdm(
                glob.iglob(
                    os.path.join(
                        label_path,
                        '*.tif')),
                disable=(
                    not self.verbose)):

                label = self.__load_single_class_from_file(path)

                labelcell.append(label)
            labelcell = np.concatenate(labelcell, axis=2)
            labelcell = np.array(labelcell, dtype=np.float16) / 255.0

            image_list.append(boxcell)
            label_list.append(labelcell)

        return image_list, label_list

    def __load_single_image_from_file(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        img = img.reshape((img.shape[0], img.shape[1], 1))

        return img

    def __load_single_class_from_file(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        b, g, r = cv2.split(img)

        r = r.reshape((r.shape[0], r.shape[1], 1))

        return r

    def split(self, train_count, val_count, val_biased=False):
        if val_biased:
            image_sample, image_val, class_sample, class_val = train_test_split(
                self.image_list, self.class_list, test_size=val_count, shuffle=True
            )
            self.image_val = image_val
            self.class_val = class_val

            unlabeled_count = image_sample.shape[0] - train_count
            image_train, image_unlabeled, class_train, class_unlabeled = train_test_split(
                image_sample, class_sample, test_size=unlabeled_count, shuffle=True)

            self.image_train = image_train
            self.class_train = class_train
            self.image_unlabeled = image_unlabeled
            self.class_unlabeled = class_unlabeled

        else:
            stratSplit = StratifiedShuffleSplit(n_splits=1, test_size=val_count)
            for sample_idxs, val_idxs in stratSplit.split(self.image_list, self.class_list):
                image_sample = self.image_list[sample_idxs]
                class_sample = self.class_list[sample_idxs]
                self.image_val = self.image_list[val_idxs]
                self.class_val = self.class_list[val_idxs]

            unlabeled_count = image_sample.shape[0] - train_count
            stratSplit = StratifiedShuffleSplit(n_splits=1, test_size=unlabeled_count)
            for train_idxs, unlabeled_idxs in stratSplit.split(image_sample, class_sample):
                self.image_train = image_sample[train_idxs]
                self.class_train = class_sample[train_idxs]
                self.image_unlabeled = image_sample[unlabeled_idxs]
                self.class_unlabeled = class_sample[unlabeled_idxs]

        if self.verbose:
            print('training : {}'.format(self.image_train.shape))
            print('validation : {}'.format(self.image_val.shape))
            print('unlabeled : {}'.format(self.image_unlabeled.shape))


    def append_train(self, add_type, append_train_count, out_dir='', model=None):
        if self.image_unlabeled is None:
            print('warning: unlabeled is None')
            return
        unlabeled_count = self.image_unlabeled.shape[0] - append_train_count
        
        if unlabeled_count >= 1:
            if add_type == 'random':
                self.__append_train_random(unlabeled_count)
            elif add_type == 'confident':
                self.__append_train_confident(append_train_count, out_dir, model)
            elif add_type == 'unconfident':
                self.__append_train_unconfident(append_train_count, out_dir, model)
            elif add_type == 'entropy':
                raise ValueError('not implemented addition type : {}'.format(add_type))
            else:
                raise ValueError('not implemented addition type : {}'.format(add_type))
        else:
            self.image_train = np.concatenate([self.image_train, self.image_unlabeled], axis=0)
            self.class_train = np.concatenate([self.class_train, self.class_unlabeled], axis=0)

            self.image_unlabeled = None
            self.class_unlabeled = None

        p = np.random.permutation(self.image_train.shape[0])
        self.image_train = self.image_train[p]
        self.class_train = self.class_train[p]
        
        if self.verbose:
            print('training : {}'.format(self.image_train.shape))
            print('validation : {}'.format(self.image_val.shape))
            if self.image_unlabeled is None:
                print('unlabeled : is None')
            else:
                print('unlabeled : {}'.format(self.image_unlabeled.shape))

    def __append_train_random(self, unlabeled_count):
        image_train, image_unlabeled, class_train, class_unlabeled = train_test_split(
            self.image_unlabeled, self.class_unlabeled, test_size=unlabeled_count, shuffle=True)
        self.image_train = np.concatenate([self.image_train, image_train], axis=0)
        self.class_train = np.concatenate([self.class_train, class_train], axis=0)

        self.image_unlabeled = image_unlabeled
        self.class_unlabeled = class_unlabeled

    def __append_train_confident(self, append_train_count, out_dir, model):
        predict_result = model.predict(
            self.image_unlabeled,
            out_dir,
        )

        # sort by top predict value
        predict_max_value = predict_result.max(axis=1).to_numpy()
        indices = np.argsort(-predict_max_value)
        image_sorted = self.image_unlabeled[indices]
        class_sorted = self.class_unlabeled[indices]

        # extract next addition value
        image_train = image_sorted[:append_train_count]
        class_train = class_sorted[:append_train_count]
        image_unlabeled = image_sorted[append_train_count:]
        class_unlabeled = class_sorted[append_train_count:]

        # overwrite _train, unlabeled
        self.image_train = np.concatenate([self.image_train, image_train], axis=0)
        self.class_train = np.concatenate([self.class_train, class_train], axis=0)
        self.image_unlabeled = image_unlabeled
        self.class_unlabeled = class_unlabeled

    def __append_train_unconfident(self, append_train_count, out_dir, model):
        predict_result = model.predict(
            self.image_unlabeled,
            out_dir,
        )

        # sort by top predict value
        predict_max_value = predict_result.max(axis=1).to_numpy()
        indices = np.argsort(predict_max_value)
        image_sorted = self.image_unlabeled[indices]
        class_sorted = self.class_unlabeled[indices]

        # extract next addition value
        image_train = image_sorted[:append_train_count]
        class_train = class_sorted[:append_train_count]
        image_unlabeled = image_sorted[append_train_count:]
        class_unlabeled = class_sorted[append_train_count:]

        # overwrite _train, unlabeled
        self.image_train = np.concatenate([self.image_train, image_train], axis=0)
        self.class_train = np.concatenate([self.class_train, class_train], axis=0)
        self.image_unlabeled = image_unlabeled
        self.class_unlabeled = class_unlabeled


    def get_one_hot(self):
        class_train = keras.utils.to_categorical(self.class_train, self.class_num)
        class_val = keras.utils.to_categorical(self.class_val, self.class_num)
        return class_train, class_val


    def input_shape(self):
        return (self.width, self.height, self.channel)

