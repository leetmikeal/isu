import numpy as np
import os

import dataset
from model import UNet

if __name__ == '__main__':
    unet = UNet(inifile='setting.ini')   
    ds = dataset.Dataset()
    
    fpath1 = unet.pre_path+'/*.tif'
    fpath2 = unet.temp3d_path+'/*.tif'
    if not os.path.exists(unet.ensemble_path):
        os.mkdir(unet.ensemble_path)
    if not os.path.exists(unet.temp_path):
        os.mkdir(unet.temp_path)
    opath = unet.temp_path + '/' + unet.dataset + '_temp_'
    
    img2d = ds.image_read(fpath1, unet.max_size, unet.max_size, feed=False)
    img3d = ds.image_read(fpath2, unet.max_size, unet.max_size, feed=False)
    ensemble = (img2d/255 + img3d/255).astype(np.uint8)
    ensemble[ensemble>0]=255
    ds.image_save(ensemble, opath)
    
