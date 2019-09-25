# -*- coding: utf-8 -*-
import glob
import os
import sys

import keras

# # adding current dir to lib path
# mydir = os.path.dirname(os.path.dirname(__file__))
# sys.path.insert(0, mydir)

import configparser
import matplotlib.pyplot as plt

class Config():
    def __init__(self, inifile='setting.ini'):
        config = configparser.SafeConfigParser()
        config.read(inifile)

        # basic directory path is the place of setting.ini
        self.__basedir = os.path.dirname(os.path.abspath(inifile))

        # path
        self.fiji_dir = self.__to_abs_path(config.get('environment', 'FIJI_DIR'))
        self.input_dir = self.__to_abs_path(config.get('environment', 'INPUT_DIR'))
        self.label_dir = self.__to_abs_path(config.get('environment', 'LABEL_DIR'))
        self.output_dir = self.__to_abs_path(config.get('environment', 'OUTPUT_DIR'))
        self.temp_dir = self.__to_abs_path(config.get('environment', 'TEMP_DIR'))
        self.dataset = config.get('environment', 'DATASET')

        # ml basic information
        self.model_2d_path = config.get('ML', 'MODEL2D')
        self.model_3d_path = config.get('ML', 'MODEL3D')
        self.max_size = config.getint('ML', 'MAX_SIZE')

        # parameter
        self.predict_batch_size = config.getint('ML.parameters', 'predict_batch_size')
        self.ke_init = 'he_normal'
             
        # generated config
        self.input_path = self.__insert_dataset(self.input_dir, self.dataset)
        self.output_path = self.__insert_dataset(self.output_dir, self.dataset)
        self.csv_path = self.__insert_dataset(self.input_dir, self.dataset, 'input.csv')


    def __to_abs_path(self, path):
        if path is None or path == '':
            return path
        return os.path.join(self.__basedir, path)

    def __insert_dataset(self, path, dataset, filename=''):
        if dataset is None or dataset == '':
            if filename is None or filename == '':
                return path
            else:
                return os.path.join(path, filename)

        else:
            if filename is None or filename == '':
                return os.path.join(path, dataset)
            else:
                return os.path.join(path, dataset, filename)




