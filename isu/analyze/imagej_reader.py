import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import imagej
from time import sleep
##import keras



# import dataset
# from model import UNet
##import tensorflow as tf

##config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.8))
##session = tf.Session(config=config)
##keras.backend.tensorflow_backend.set_session(session)



def Process3DOC(ij, open_init_file, savedir, savefile):
    """3D Object Counter
    
    Parameters
    ----------
    ij : imagej object

    unet : UNet object
        defined in model.py

    """
    # openfile = unet.pre_path + '/' + unet.dataset + '_RB_0000.tif'
    # savedir = unet.output_path
    # savefile = unet.dataset + '_label_'
    print('open: {}'.format(open_init_file))
    print('savedir: {}'.format(savedir))
    print('savefile: {}'.format(savefile))
    
    

    print('1.Open Image Sequence')
    plugin = 'Image Sequence...'
    args = {'open': open_init_file,
            'number': '300',
            'increment': '1',
            'scale': '100',
            'sort': True,}
    ij.py.run_plugin(plugin, args, ij1_style=True)

    
    print('2.Running 3DOC Options')
    plugin = '3D OC Options'
    args = {'close_original_images_while_processing_(saves_memory)': True,
            'volume': True,
            'nb_of_obj._voxels': True,
            'centroid': True,}
    ij.py.run_plugin(plugin, args, ij1_style=True)
    
    print('3.Running 3D Objects Counter(take a few minutes...)')
    plugin = '3D Objects Counter'
    args = {'threshold': '70',
            'slice': '10',
            'min.': '1',
            'max.': '4000000',
            'objects': True,
            'statistics': True,
            'summary': True,}
    ij.py.run_plugin(plugin, args, ij1_style=True)
    
    print('4.Save Image Sequence')
    plugin = 'Image Sequence... '
    args = {'save': savedir,
            'format': 'TIFF',
            'name': savefile,}
    ij.py.run_plugin(plugin, args, ij1_style=True)
    
    macro =  'saveAs("Results", "' + savedir + '.csv"); '
    ij.script().run('macro.ijm', macro, True).get()
    
    #sleep(1000)
    print('5.All Process is Done.')
    ij.script().run('macro.ijm', 'run("Close All");', True).get()
    

def main():
    unet = UNet(inifile='setting.ini')
    ds = dataset.Dataset()
    
    fiji_path = os.path.join(unet.fiji_dir)
    ij = imagej.init(fiji_path, headless=False)
    print(ij.getVersion())
    ij.batchmode = True
    #ij.ui().showUI()
   
   
    if not os.path.exists(unet.output_path):
        os.mkdir(unet.output_path)

    Process3DOC(ij, unet)
   
if __name__ == '__main__':
    main()

