import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import cc3d
import cv2


def setup_argument_parser(parser):
    """
    Set argument
    """
    parser.add_argument('--in-dir', help='image contained directory path', required=True)
    parser.add_argument('--out-dir', help='path', required=True)
    parser.add_argument('--in-filename', help='filename', default='*.tif')
    parser.add_argument('--connectivity', help='neighbor search area in 3d box. [26, 18, 6]', type=int, default=6)
    parser.add_argument('--stat', help='statistics information saving file path')
    parser.add_argument('--verbose', help='output process detail', action='store_true')



def analyze_connection(in_dir, out_dir, in_filename='*.tif', connectivity=6, stat=None, verbose=False):
    if not os.path.exists(in_dir):
        raise ValueError('input directory was not found. | {}'.format(in_dir))
    if connectivity not in [6, 18, 26]:
        raise ValueError('wrong connectivity value was set. | {}'.format(connectivity))

    # loading
    box = load_box(in_dir, in_filename)

    # connection
    box = box.astype(np.int32)
    connected = cc3d.connected_components(box, connectivity=connectivity)

    # statistics
    if stat is not None:
        process_stat(connected, stat, verbose)

    # save
    os.makedirs(out_dir, exist_ok=True)
    save_box(connected, out_dir, verbose)

    if verbose:
        print('completed')

def process_stat(connected, path, verbose=False):
    # prepare directory
    parent_dir = os.path.dirname(path)
    os.makedirs(parent_dir, exist_ok=True)

    # process
    n_region, volumes = get_stat(connected)

    with open(path, 'w') as f:
        f.writelines('region count: {}'.format(n_region) + '\n')
        f.writelines('volumes:' + '\n')
        for i, v in enumerate(volumes):
            f.writelines('{:>5d} : {}'.format(i, v) + '\n')

    if verbose:
        print('region count: {}'.format(n_region))
        print('volumes:')
        for i, v in enumerate(volumes):
            print('{:>5d} : {}'.format(i, v))


    
def get_stat(connected):
    n_region = connected.max()

    # area size
    # volumes = [0] * n_region
    # for i in tqdm(range(n_region)):
    #     v = np.count_nonzero(connected == i + 1)
    #     volumes.append(v)

    flat = connected.flatten()
    sr = pd.Series(flat).value_counts(sort=False)

    volumes = sr.tolist()[1:]


    return n_region, volumes


def load_box(path, filename, verbose=False):
    if verbose:
        print('image loading. | {}'.format(path))
    
    image_list = []
    for p in tqdm(glob.glob(os.path.join(path, filename))):
        img = cv2.imread(p)
        gray_img = np.zeros(img.shape[:2], dtype=np.uint8)
        gray_img[img[:,:,0] > 0] = 1
        gray_img[img[:,:,1] > 0] = 1
        gray_img[img[:,:,2] > 0] = 1
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
        b = box[:, :, k]

        img = convert_color_box(b)

        path = os.path.join(out_dir, '{:04d}.tif'.format(k))
        cv2.imwrite(path, img)

def convert_color_box(img32):
    r = pickup_byte0(img32, 0)
    g = pickup_byte1(img32, 1)
    b = pickup_byte2(img32, 2)
    img = np.concatenate((b, g, r), axis=2)
    return img

def pickup_byte0(img32, channel):
    arr = img32.astype(np.uint8).reshape(img32.shape + (1,))
    return arr

def pickup_byte1(img32, channel):
    def shift(b):
        return b >> 8

    flat = img32.flatten()
    f = np.frompyfunc(shift, 1, 1)
    arr = f(flat)
    arr = arr.astype(np.uint8).reshape(img32.shape + (1,))
    return arr

def pickup_byte2(img32, channel):
    def shift(b):
        return b >> 16

    flat = img32.flatten()
    f = np.frompyfunc(shift, 1, 1)
    arr = f(flat)
    arr = arr.astype(np.uint8).reshape(img32.shape + (1,))
    return arr
    



def main(args):
    analyze_connection(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        in_filename=args.in_filename,
        stat=args.stat,
        connectivity=args.connectivity,
        verbose=args.verbose
    )


if __name__ == '__main__':
    test_args = type("Hoge", (object,), {
        'save_dir': 'work/mnist',
        'verbose': True,
    })
    main(test_args)
