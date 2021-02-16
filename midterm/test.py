import sklearn
from sklearn import preprocessing
from datetime import datetime
import numpy as np
import os
import re
import tensorflow as tf
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
default_root = 'data/'
csv = '.*\.csv'

def ls(path, type=None):
    if type is None: return [os.path.join(path, name) for name in os.listdir(path)]
    paths = []
    for name in os.listdir(path):
        if (re.match(type, name)):
            paths.append(os.path.join(path, name))
    return paths

test_set = np.load(default_root + 'test.npz')
test_x = test_set['inputs'].astype(np.float)
test_y = test_set['targets'].astype(np.int)

model=tf.keras.models.load_model(default_root+'model/best.h5')
model.summary()
x=test_x[0].shape
print(x)
x=model.predict(test_x[0])
x=sklearn.metrics.confusion_matrix(test_y,test_x)

