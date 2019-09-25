import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

from core import Config


def setup_argument_parser(parser):
    """
    Set argument
    """
    parser.add_argument(
        '--in-settings',
        help='set setting file path [settings.ini]',
        required=True)
    parser.add_argument(
        '--dataset',
        help='overwrite dataset name in settings.ini by command',
        default=None)
    # parser.add_argument('--in-dir1', help='image contained directory path 1', required=True)
    # parser.add_argument('--in-dir2', help='image contained directory path 2', required=True)
    # parser.add_argument('--out-dir', help='output directory path', required=True)
    parser.add_argument('--verbose', help='output process detail', action='store_true')


def analyze_ensemble(in_dir1, in_dir2, out_dir, verbose):
    if not os.path.exists(in_dir1):
        raise ValueError('input directory was not found. | {}'.format(in_dir1))
    elif not os.path.exists(in_dir2):
        raise ValueError('input directory was not found. | {}'.format(in_dir2))

    # unet = UNet(inifile='setting.ini')
    # ds = dataset.Dataset()

    fpath1 = os.path.join(in_dir1, '*.tif')
    fpath2 = os.path.join(in_dir2, '*.tif')
    os.makedirs(out_dir, exist_ok=True)

    volume1 = image_read(fpath1)
    volume2 = image_read(fpath2)
    ensemble = (volume1 / 255 + volume2 / 255).astype(np.uint8)
    ensemble[ensemble > 0] = 255

    image_save(ensemble, out_dir)


def image_read(dirpath):
    """Read tiff images

    Parameters
    ----------
    dirpath : str
        image file path

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

    filepaths = sorted([p for p in glob.glob(dirpath)])

    image_list = []
    for path in filepaths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = img.reshape(img.shape + (1,))
        image_list.append(img)

    volume = np.concatenate(image_list, axis=2)
    return volume


def image_save(image, out_dir):
    """Save image stack as TIFF files

    Parameters
    ----------
    image : numpy array shape = (width, height, number)
        image stack
    out_dir : str
        filename path of output

    """
    for k in range(image.shape[2]):
        path = os.path.join(out_dir, '{:04d}.tif'.format(k))
        cv2.imwrite(path, image[:, :, k])


def main(args):
    config = Config(args.in_settings)
    config.init(args.dataset)
    if args.verbose:
        config.debug()

    in_dir1 = config.temp_2d_dir
    in_dir2 = config.temp_3d_dir
    out_dir = config.temp_ensemble_dir

    analyze_ensemble(
        in_dir1=in_dir1,
        in_dir2=in_dir2,
        out_dir=out_dir,
        verbose=args.verbose
    )


if __name__ == '__main__':
    # test_args = type("Hoge", (object,), {
    #     'save_dir': 'work/mnist',
    #     'verbose': True,
    # })
    # main(test_args)

    a = np.array([0, 1, 0, 1])
    b = np.array([0, 1, 1, 0])
    count = count_pixel_confusion_matrix(a, b)
    print(count)
