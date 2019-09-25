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

        self.fiji_dir = self.__to_abs_path(config.get('environment', 'FIJI_DIR'))
        self.input_dir = self.__to_abs_path(config.get('environment', 'INPUT_DIR'))
        self.output_dir = self.__to_abs_path(config.get('environment', 'OUTPUT_DIR'))
        self.temp_dir = self.__to_abs_path(config.get('environment', 'TEMP_DIR'))
        self.dataset = config.get('environment', 'DATASET')

        self.model_name = config.get('ML', 'MODEL2D_NAME')
        self.max_size = config.getint('ML', 'MAX_SIZE')
        self.predict_batch_size = config.getint('ML.parameters', 'predict_batch_size')
             
        self.input_path = self.input_dir + self.dataset
        self.output_path = self.output_dir + self.dataset
        self.temp2d_path = self.temp_dir + 'temp2d/'
        self.pre_path = self.temp2d_path + self.dataset
        self.temp3d_path = self.temp_dir + 'temp3d/' + self.dataset
        self.ensemble_path = self.temp_dir + 'ensemble/'
        self.temp_path = self.ensemble_path + self.dataset
        self.csv_path = self.input_dir + self.dataset + '/input.csv'
        self.ke_init = 'he_normal'

    def __to_abs_path(self, path):
        if path is None or path == '':
            return path
        return os.path.join(self.__basedir, path)



