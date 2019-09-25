import glob
import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import cc3d
from core import Config


def setup_argument_parser(parser):
    """
    Set argument
    """
    parser.add_argument('--in-dir', help='image contained directory path', required=True)
    parser.add_argument('--out-dir', help='path', required=True)
    parser.add_argument('--in-filename', help='filename', default='*.tif')
    parser.add_argument('--threshold', help='', type=int, default=1)
    parser.add_argument('--verbose', help='output process detail', action='store_true')


def analyze_connection(in_dir, out_dir, in_filename, threshold=1, verbose=False):
    if not os.path.exists(in_dir):
        raise ValueError('input directory was not found. | {}'.format(in_dir))

    # loading
    box = load_box(in_dir, in_filename, threshold=threshold)

    # save
    os.makedirs(out_dir, exist_ok=True)
    save_box(box, out_dir, verbose)

    if verbose:
        print('completed')


def load_box(path, filename, threshold=1, verbose=False):
    if verbose:
        print('image loading. | {}'.format(path))

    image_list = []
    for p in tqdm(glob.glob(os.path.join(path, filename)), disable=(not verbose)):
        img = cv2.imread(p)
        gray_img = np.zeros(img.shape[:2], dtype=np.uint8)
        gray_img[img[:, :, 0] >= threshold] = 255
        gray_img[img[:, :, 1] >= threshold] = 255
        gray_img[img[:, :, 2] >= threshold] = 255
        gray_img = gray_img.reshape(gray_img.shape + (1,))
        image_list.append(gray_img)

    if len(image_list) == 0:
        raise ValueError('image file was not found.')

    # return np.array(image_list)
    boxcell = np.concatenate(image_list, axis=2)
    return boxcell


def save_box(box, out_dir, verbose=False):
    """Save image stack as TIFF files

    Parameters
    ----------
    box : numpy array shape = (width, height, number)
        image stack
    out_dir : str
        filename path of output

    """
    for k in tqdm(range(box.shape[2]), disable=(not verbose)):
        img = box[:, :, k]
        path = os.path.join(out_dir, '{:04d}.tif'.format(k))
        cv2.imwrite(path, img)


def main(args):
    analyze_connection(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        in_filename=args.in_filename,
        threshold=args.threshold,
        verbose=args.verbose
    )


if __name__ == '__main__':
    test_args = type("Hoge", (object,), {
        'save_dir': 'work/mnist',
        'verbose': True,
    })
    main(test_args)
