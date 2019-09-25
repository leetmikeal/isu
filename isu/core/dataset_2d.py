import sys
import os
import pylab
import glob
import numpy as np
from PIL import Image


class Dataset2d:
    """ Dataset Class

    Parameters
    ----------
    classes : int (default: 1)
        number of class labels

    """

    def __init__(self, classes=1):
        self.classes = classes

    def image_read(self, file_path, width, height, feed=True):
        """Read tiff images

        Parameters
        ----------
        file_path : str 
            image file path
        width : int
            width of image
        height : int
            height of image
        feed : bool
            flag of changing the data's shape for feeding network
            if True, (width, height, channel, num)->(num, width, height, channel)

        Attribute
        ---------
        filenames : list
            name list of image files
        temp : image
            one sample of image files
        channel: int (default:0)
            grayscale:1, full color:3

        Returns
        -------
        volume : np.array, shape = (width, height, channels, len(filenames))
            image stack

        """

        filenames = [img for img in glob.glob(file_path)]
        filenames.sort()
        temp = pylab.imread(filenames[0])
        h = len(filenames)
        channel = 0

        if len(temp.shape) == 2:
            d, w = temp.shape
            volume = np.zeros((w, d, h), dtype=np.uint8)
            channel = 1

        elif len(temp.shape) == 3:
            d, w, c = temp.shape
            volume = np.zeros((w, d, c, h), dtype=np.uint8)
            channel = 3

        k = 0
        for img in filenames:  # assuming tif
            im = pylab.imread(img)
            if channel == 1:
                #assert im.shape == (width, height), 'Image with an unexpected size'
                volume[:, :, k] = im[:w, :d]
            elif channel == 3:
                #assert im.shape == (width, height, 3), 'Image with an unexpected size'
                volume[:, :, :, k] = im[:w, :d, :]
            k += 1

        # rollaxis change for Keras
        if feed:
            if channel == 1:
                volume = np.rollaxis(volume, axis=2, start=0)
            else:
                volume = np.rollaxis(volume, axis=3, start=0)
        return volume

    def load_csv(self, csvPath):
        """Load setting csv file

        Parameters
        ----------
        csvPath : str 
            the path of "input.csv"

        Returns
        -------
        width, height and number of images

        """

        with open(csvPath) as f:
            lines = [s.strip() for s in f.readlines()]
        return int(lines[2]), int(lines[3]), int(lines[1])

    def color_threshold(self, img, th_value=0):
        """Load setting csv file

        Parameters
        ----------
        img : np.array shape = (width, height, channel, num) 
            color image stack
        th_value : int
            cutting threshold for grayscale value

        Returns
        -------
        volume : np.array shape = (width, height, num)
            binary image stack, the value of each pixel is 0 or 255

        """

        weights = np.c_[0.2989, 0.5870, 0.1140]

        if len(img.shape) == 3:
            tile = np.tile(weights, reps=(img.shape[0], img.shape[1], 1))
            return (np.sum(tile * img, axis=2) > th_value) * 255

        else:
            volume = np.zeros((img.shape[0], img.shape[1], img.shape[3]), dtype=np.uint8)
            for k in range(img.shape[3]):
                im = img[:, :, :, k]
                tile = np.tile(weights, reps=(im.shape[0], im.shape[1], 1))
                volume[:, :, k] = (np.sum(tile * im, axis=2) > th_value) * 255
            return volume

    def add_rotdata(self, X, Y):
        X90 = np.zeros(X.shape, dtype=np.float32)
        Y90 = np.zeros(Y.shape, dtype=np.float32)
        X180 = np.zeros(X.shape, dtype=np.float32)
        Y180 = np.zeros(Y.shape, dtype=np.float32)
        X270 = np.zeros(X.shape, dtype=np.float32)
        Y270 = np.zeros(Y.shape, dtype=np.float32)

        for k in range(X.shape[0]):
            X90[k, :, :] = np.rot90(X[k])
            Y90[k, :, :] = np.rot90(Y[k])
            X180[k, :, :] = np.rot90(X[k], 2)
            Y180[k, :, :] = np.rot90(Y[k], 2)
            X270[k, :, :] = np.rot90(X[k], 3)
            Y270[k, :, :] = np.rot90(Y[k], 3)
        return np.concatenate([X, X90, X180, X270]), np.concatenate([Y, Y90, Y180, Y270])

    def image_save(self, image, fname):
        """Save image stack as TIFF files

        Parameters
        ----------
        image : numpy array shape = (width, height, number) 
            image stack
        fname : str
            filename path of output

        """
        for k in range(image.shape[2]):
            n = ('000' + str(k))[-4:]
            Image.fromarray(image[:, :, k]).save(fname + n + '.tif')

        return fname + n + '.tif'
