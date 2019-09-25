import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

def setup_argument_parser(parser):
    """
    Set argument
    """
    parser.add_argument('--in-dir1', help='image contained directory path 1', required=True)
    parser.add_argument('--in-dir2', help='image contained directory path 2', required=True)
    parser.add_argument('--out-csv', help='output csv file path', required=True)
    parser.add_argument('--verbose', help='output process detail', action='store_true')



def analyze_pxcm(in_dir1, in_dir2, out_csv, verbose):
    if not os.path.exists(in_dir1):
        raise ValueError('input directory was not found. | {}'.format(in_dir1))
    elif not os.path.exists(in_dir2):
        raise ValueError('input directory was not found. | {}'.format(in_dir2))

    box1 = load_box(in_dir1)
    box2 = load_box(in_dir2)

    tp, fp, fn, tn = count_pixel_confusion_matrix(box1, box2)

    matrix = [[tp, fp], [fn, tn]]

    import pandas as pd
    df = pd.DataFrame(matrix, columns=['pred_pos', 'pred_neg'], index=['truth_pos', 'truth_neg'])
    df.to_csv(out_csv)
    


def count_pixel_confusion_matrix(box1, box2):
    if box1.shape != box2.shape:  # compare full shape
        raise ValueError('not match shape| {} : {}'.format(box1.shape, box2.shape))

    flat1 = box1.flatten()
    flat2 = box2.flatten()

    flat1_nonzero = flat1 != 0
    flat1_zero = flat1 == 0
    flat2_nonzero = flat2 != 0
    flat2_zero = flat2 == 0

    tp = np.sum(np.logical_and(flat1_nonzero, flat2_nonzero))
    fp = np.sum(np.logical_and(flat1_nonzero, flat2_zero))
    fn = np.sum(np.logical_and(flat1_zero, flat2_nonzero))
    tn = np.sum(np.logical_and(flat1_zero, flat2_zero))

    return tp, fp, fn, tn



def load_box(path, verbose=False):
    if verbose:
        print('image loading. | {}'.format(path))
    
    image_list = []
    for p in tqdm(glob.glob(os.path.join(path, '*.tif'))):
        img = cv2.imread(p)
        gray_img = np.zeros(img.shape[:2], dtype=np.uint8)
        gray_img[img[:,:,0] > 0] = 1
        gray_img[img[:,:,1] > 0] = 1
        gray_img[img[:,:,2] > 0] = 1
        gray_img = gray_img.reshape(gray_img.shape + (1,))
        image_list.append(gray_img)

    # return np.array(image_list)
    full_array = boxcell = np.concatenate(image_list, axis=2)
    return full_array



def main(args):
    analyze_pxcm(
        in_dir1=args.in_dir1,
        in_dir2=args.in_dir2,
        out_csv=args.out_csv,
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
