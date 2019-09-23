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
    def __init__(self, dir_path, cache_image, crop_size, data_count, verbose):
        self.dir_path = dir_path
        self.cache_image = cache_image
        self.crop_size = crop_size
        self.data_count = data_count
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

                self.image_list = image_list
                self.class_list = class_list
                self.__set_attributes()

                if self.verbose:
                    self.debug_print()
                    
                return

        image_raw_list, class_raw_list = self.__load_image(self.dir_path)
        image_list, class_list = self.__crop_image(image_raw_list, class_raw_list, self.data_count, self.crop_size)

        if self.cache_image is not None and os.path.exists(self.cache_image):
            image_cache_path, class_cache_path = self.__get_cache_file_path()
            np.save(image_cache_path, image_list, allow_pickle=False)
            np.save(class_cache_path, class_list, allow_pickle=False)

        self.image_list = image_list
        self.class_list = class_list
        self.__set_attributes()

        if self.verbose:
            self.debug_print()

    def debug_print(self):
        return
        for idx, [i, c] in enumerate(zip(self.image_list, self.class_list)):
            print('loaded {} image : {}'.format(idx, i.shape))
            print('loaded {} class : {}'.format(idx, c.shape))



    def __set_attributes(self):
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
            boxcell = np.array(boxcell, dtype=np.float32)

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
            labelcell = np.array(labelcell, dtype=np.float32)

            image_list.append(boxcell)
            label_list.append(labelcell)

        return image_list, label_list


    def __crop_image(self, image_raw_list, class_raw_list, count, size):
        if self.verbose:
            print('cropping image')

        raw_count = len(image_raw_list)
        # image_index_list = np.random.randint(0, raw_count, count)

        crop_image_list = []
        crop_class_list = []
        c = 0
        while(c < count):
            idx = np.random.randint(0, raw_count)
            image = image_raw_list[idx]
            cl = class_raw_list[idx]

            range_x = np.random.randint(0, image.shape[0] - size[0])
            range_y = np.random.randint(0, image.shape[1] - size[1])
            range_z = np.random.randint(0, image.shape[2] - size[2])

            crop_image = image[range_x:range_x+size[0], range_y:range_y+size[1], range_z:range_z+size[2], :]
            crop_class = cl[range_x:range_x+size[0], range_y:range_y+size[1], range_z:range_z+size[2], :]

            if (crop_class.max() == 0):
                continue

            crop_image_list.append(crop_image)
            crop_class_list.append(crop_class)
            c += 1

        return np.array(crop_image_list), np.array(crop_class_list)

        

    def __load_single_image_from_file(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        img = img.reshape((img.shape[0], img.shape[1], 1, 1))

        img = img.astype(np.float32)
        img -= np.mean(img, keepdims=True)
        img /= (np.std(img, keepdims=True) + 1e-6)

        # img = img.astype(np.float32) / 255.0 - 0.5

        return img

    def __load_single_class_from_file(self, path):
        img_color = cv2.imread(path, cv2.IMREAD_COLOR)
        b, g, r = cv2.split(img_color)

        img = np.zeros((img_color.shape[0], img_color.shape[1]), dtype=np.int8)
        img[r > 0] = 1
        img[g > 0] = 1
        img[b > 0] = 1
        img = img.reshape((img.shape[0], img.shape[1], 1, 1))
        # img = img.astype(np.float32) / 255.0

        return img

    def split(self, train_count, val_count, val_biased=False):
        if val_biased:
            image_train, image_val, class_train, class_val = train_test_split(
                self.image_list, self.class_list, test_size=val_count, shuffle=True
            )
            self.image_val = image_val
            self.class_val = class_val

            self.image_train = image_train
            self.class_train = class_train

        else:
            stratSplit = StratifiedShuffleSplit(n_splits=1, test_size=val_count)
            for sample_idxs, val_idxs in stratSplit.split(self.image_list, self.class_list):
                image_sample = self.image_list[sample_idxs]
                class_sample = self.class_list[sample_idxs]
                self.image_val = self.image_list[val_idxs]
                self.class_val = self.class_list[val_idxs]

            self.image_train = image_sample
            self.class_train = class_sample

        if self.verbose:
            print('training : {}'.format(self.image_train.shape))
            print('validation : {}'.format(self.image_val.shape))



    def input_shape(self):
        return (self.width, self.height, self.channel, 1)

