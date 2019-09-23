# -*- coding: utf-8 -*-
import glob
import os
import math
import sys

import cv2
import keras
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from tqdm import tqdm


class SampleSingle():
    def __init__(self, image_dir_path, verbose):
        self.image_dir_path = image_dir_path
        self.verbose = verbose

        self.image = None
        self.width = 0
        self.height = 0
        self.depth = 0
        self.padding_position = None

        self.load()

    def load(self):
        # if self.cache_image is not None:
        #     image_cache_path, class_cache_path = self.__get_cache_file_path()
        #     if os.path.exists(image_cache_path) and os.path.exists(
        #             class_cache_path):
        #         image_list = np.load(image_cache_path, allow_pickle=False)
        #         class_list = np.load(class_cache_path, allow_pickle=False)

        #         self.image_list = image_list
        #         self.__set_attributes(self.image)

        #         if self.verbose:
        #             self.debug_print()

        #         return

        image_raw = self.__load_image(self.image_dir_path)
        self.image = image_raw


    def debug_print(self):
        return
        for idx, [i, c] in enumerate(zip(self.image_list, self.class_list)):
            print('loaded {} image : {}'.format(idx, i.shape))
            print('loaded {} class : {}'.format(idx, c.shape))

    def __set_attributes(self, image):
        self.width = image.shape[0]
        self.height = image.shape[1]
        self.depth = image.shape[2]

    # def __get_cache_file_path(self):
    #     image_path = os.path.join(self.cache_image, 'image.npy')
    #     class_path = os.path.join(self.cache_image, 'class.npy')
    #     return image_path, class_path

    def __load_image(self, image_dir_path):
        """load images

        Args:
            image_dir_path (string): loadimage base directory path

        Returns:
            image_list: image list by numpy array
        """
        def get_class_name(path):
            dirname = os.path.basename(os.path.dirname(path))
            return dirname

        csv_path = os.path.join(image_dir_path, 'input.csv')
        name, z, x, y = self.__load_csv(csv_path)

        boxcell = []
        for filename in tqdm(self.__generate_file_name(name, z), disable=(not self.verbose)):
            path = os.path.join(image_dir_path, filename)

            img = self.__load_single_image_from_file(path)
            boxcell.append(img)

        boxcell = np.concatenate(boxcell, axis=2)
        boxcell = np.array(boxcell, dtype=np.float32)

        # set size
        self.__set_attributes(boxcell)

        # padding
        x_front = int(math.ceil((400 - self.width) / 2))
        x_back = int(math.floor((400 - self.width) / 2))
        y_front = int(math.ceil((400 - self.height) / 2))
        y_back = int(math.floor((400 - self.height) / 2))
        z_front = int(math.ceil((400 - self.depth) / 2))
        z_back = int(math.floor((400 - self.depth) / 2))
        self.padding_position = (x_front, y_front, z_front)
        #boxcell = np.pad(boxcell, ((x_front, x_back), (y_front, y_back), (z_front, z_back), (0, 0)), 'constant')
        boxcell = np.pad(boxcell, ((x_front, x_back), (y_front, y_back), (z_front, z_back), (0, 0)), 'edge')

        return boxcell


    def __load_csv(self, path):
        with open(path, 'r') as f:
            name = f.readline().strip('\n')
            z = int(f.readline())
            x = int(f.readline())
            y = int(f.readline())

        return name, z, x, y


    def __generate_file_name(self, name, z_count):
        for i in range(z_count):
            yield '{}_input_{:04d}.tif'.format(name, i)


    def __load_single_image_from_file(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        img = img.reshape((img.shape[0], img.shape[1], 1, 1))

        img = img.astype(np.float32)
        img -= np.mean(img, keepdims=True)
        img /= (np.std(img, keepdims=True) + 1e-6)

        # img = img.astype(np.float32) / 255.0 - 0.5

        return img

    def originl_input_shape(self):
        return (self.width, self.height, self.depth, 1)

    def input_shape(self):
        return self.image.shape
