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
    parser.add_argument('--in-dir', help='image contained directory path', required=True)
    parser.add_argument('--out-csv', help='csv output path', required=True)
    parser.add_argument('--verbose', help='output process detail', action='store_true')


def region_count(in_dir, out_csv, verbose=False):
    if not os.path.exists(in_dir):
        raise ValueError('input directory was not found. | {}'.format(in_dir))

    fpath = os.path.join(in_dir, '*.tif')

    voxel = voxel_read(fpath, verbose)
    if verbose:
        print('input shape : {}'.format(voxel.shape))

    result_count = count_collision(voxel, verbose)

    save_result(result_count, out_csv)
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


def count_collision(voxel, verbose=False):
    flat = voxel.flatten()

    n_region = flat.max()
    result = []
    for segid in tqdm(range(1, n_region + 1), disable=(not verbose)):
        count = np.count_nonzero(flat == segid)
        result.append([segid, count])

    return result


def save_result(result, path):
    np_result = np.array(result)
    data = np_result[:, 1:]
    index = np_result[:, 0]
    print('data : {}'.format(data.shape))
    print('index : {}'.format(index.shape))
    df = pd.DataFrame(data, index=index, columns=['volume'])
    df.sort_values('volume', ascending=False, inplace=True)
    df.index
    df.to_csv(path)



def main(args):
    region_count(
        in_dir=args.in_dir,
        out_csv=args.out_csv,
        verbose=args.verbose
    )


if __name__ == '__main__':
    test_args = type("Hoge", (object,), {
        'save_dir': 'work/mnist',
        'verbose': True,
    })
    main(test_args)
