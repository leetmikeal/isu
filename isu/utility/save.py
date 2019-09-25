import os

import numpy as np
from keras.datasets import mnist
from PIL import Image


def check_dir(path):
    """checking to exist directory and create
    
    Args:
        path (string): checking directory path
    """
    dirpath = os.path.dirname(path)
    os.makedirs(dirpath, exist_ok=True)


def save(save_dir, data, index, num):
    """saving image array to file
    
    Args:
        save_dir (string): saving directory path
        data (numpy.array): image data
        index (int): image index
        num (int): class number
    """
    img = Image.fromarray(data)

    filename = os.path.join(
        save_dir,
        str(num),
        '{0:08d}'.format(index) +
        '.png')
    check_dir(filename)

    # img_resized = img.resize((280, 280))
    img.save(filename)


