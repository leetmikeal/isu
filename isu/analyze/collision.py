import glob
import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def setup_argument_parser(parser):
    """
    Set argument
    """
    parser.add_argument('--in-dir1', help='image contained directory path 1', required=True)
    parser.add_argument('--in-dir2', help='image contained directory path 2', required=True)
    parser.add_argument('--out-csv', help='output csv path', required=True)
    parser.add_argument('--verbose', help='output process detail', action='store_true')


def analyze_collision(in_dir1, in_dir2, out_csv, verbose):
    if not os.path.exists(in_dir1):
        raise ValueError('input directory was not found. | {}'.format(in_dir1))
    elif not os.path.exists(in_dir2):
        raise ValueError('input directory was not found. | {}'.format(in_dir2))

    # unet = UNet(inifile='setting.ini')
    # ds = dataset.Dataset()

    fpath1 = os.path.join(in_dir1, '*.tif')
    fpath2 = os.path.join(in_dir2, '*.tif')

    voxel1 = voxel_read(fpath1, verbose)
    voxel2 = voxel_read(fpath2, verbose)
    if verbose:
        print('input1 shape : {}'.format(voxel1.shape))
        print('input2 shape : {}'.format(voxel2.shape))
    if voxel2.shape != voxel2.shape:
        raise ValueError('not match each input shape')

    result1_count = count_collision(voxel1, voxel2, verbose)
    result2_count = count_collision(voxel2, voxel1, verbose)

    save_result(result1_count, result2_count, out_csv)
    if verbose:
        print('completed!')


def voxel_read(search_path, verbose=False):
    """Read tiff images to convert to voxel data

    Parameters
    ----------
    search_path : str
        image file path

    Returns
    -------
    volume : np.array, shape = (width, height, channels, len(filenames))
        image stack

    """

    filepaths = sorted([p for p in glob.glob(search_path)])

    image_list = []
    for path in tqdm(filepaths, disable=(not verbose)):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = sequence_region_index(img)
        img = img.reshape(img.shape + (1,))
        image_list.append(img)

    volume = np.concatenate(image_list, axis=2)
    return volume


def sequence_region_index(img):
    """3 channel image to single channel
    """
    b, g, r = cv2.split(img)
    r = r.astype(np.uint32)
    g = np.left_shift(g.astype(np.uint32), 8)
    b = np.left_shift(b.astype(np.uint32), 16)
    new_img = r + g + b
    return new_img


def count_collision(voxel1, voxel2, verbose=False):
    flat1 = voxel1.flatten()
    flat2 = voxel2.flatten()

    n_region = flat1.max()
    col_count = 0
    uncol_count = 0
    for segid in tqdm(range(1, n_region + 1), disable=(not verbose)):
        extracted = flat2 * (flat1 == segid)
        if np.any(extracted > 0):
            col_count += 1
        else:
            uncol_count += 1

    return col_count, uncol_count, n_region


def save_result(result1, result2, out_csv):
    dirpath = os.path.dirname(os.path.abspath(out_csv))
    os.makedirs(dirpath, exist_ok=True)

    data = [result1, result2]
    df = pd.DataFrame(data, index=['voxel1', 'voxel2'], columns=['collision', 'avoidance', 'total'])
    df.to_csv(out_csv)


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
    in_dir1 = args.in_dir1
    in_dir2 = args.in_dir2
    out_csv = args.out_csv

    analyze_collision(
        in_dir1=in_dir1,
        in_dir2=in_dir2,
        out_csv=out_csv,
        verbose=args.verbose
    )


if __name__ == '__main__':
    # test_args = type("Hoge", (object,), {
    #     'save_dir': 'work/mnist',
    #     'verbose': True,
    # })
    # main(test_args)
    print('main')

