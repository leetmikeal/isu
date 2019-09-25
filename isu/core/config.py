# -*- coding: utf-8 -*-
import configparser
import glob
import os
import sys


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
        self.prefix_2d = config.get('environment', 'PREFIX_2D')
        self.prefix_3d = config.get('environment', 'PREFIX_3D')

        # ml basic information
        self.model_2d_path = config.get('ML', 'MODEL2D')
        self.model_3d_path = config.get('ML', 'MODEL3D')
        self.max_size = config.getint('ML', 'MAX_SIZE')

        # parameter
        self.predict_2d_batch_size = config.getint('ML.parameters', 'PREDICT_2D_BATCH_SIZE')
        self.ke_init = 'he_normal'

    def __to_abs_path(self, path):
        if path is None or path == '':
            return path
        return os.path.join(self.__basedir, path)

    def __insert_dataset(self, path, dataset):
        if dataset is None or dataset == '':
            return path
        else:
            return os.path.join(path, dataset)

    def init(self, dataset=None):
        used_dataset = dataset
        if used_dataset is None:
            used_dataset = self.dataset

        # generated config
        self.temp_2d_dir = self.__insert_dataset(os.path.join(self.temp_dir, self.prefix_2d), used_dataset)
        self.temp_3d_dir = self.__insert_dataset(os.path.join(self.temp_dir, self.prefix_3d), used_dataset)
        self.input_path = self.__insert_dataset(self.input_dir, used_dataset)
        self.output_path = self.__insert_dataset(self.output_dir, used_dataset)

        self.csv_path = os.path.join(self.__insert_dataset(self.input_dir, used_dataset), 'input.csv')
