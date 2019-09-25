# isu

Isu prediction in 3d boxcell data


## data preparing

Put data follow `work/` directory.

 - work/
   - input/
     - alcon01/
     - alcon06/
     - alcon07/
     - alcon08/
     - alcon12/
   - label/
     - alcon01/
     - alcon06/
     - alcon07/
     - alcon08/
     - alcon12/

## installation

To install tensorflow and Keras, you have to manually run follow commands. Because of select `tensorflow` or `tensorflow-gpu`. Recommended `tensorflow==1.14` and `keras==2.2`.

```
pip install tensorflow==1.14  # or use gpu, tensorflow-gpu==1.14
pip install keras
```

Next, install this repository package.

```
pip install -e .
```

or 

```
pip install -r requirements.txt
```

Then, you can use `isu` command or `python isu/isu.py`.


## execution

see `scripts/predict.sh`

4 steps:

 - predict from 2D model by `isu predict-2d`
 - predict from 3D model by `isu predict-3d`
 - make ensemble result by `isu analyze ensemble`
 - blob process by `isu analyze connection`


## settings

Some parameter can changed in each environment to edit `setting.ini`.
