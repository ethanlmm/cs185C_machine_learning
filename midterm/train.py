import numpy as np
import tensorflow as tf
from tensorflow import keras
from model import  *
import os

from datetime import datetime
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

default_root = 'data/'
default_model_save_root=default_root+"model/"

train_set = np.load(default_root + 'train.npz')
train_x = train_set['inputs'].astype(np.float)
train_y = train_set['targets'].astype(np.int)

val_set = np.load(default_root + 'val.npz')
val_x = val_set['inputs'].astype(np.float)
val_y = val_set['targets'].astype(np.int)

test_set = np.load(default_root + 'test.npz')
test_x = test_set['inputs'].astype(np.float)
test_y = test_set['targets'].astype(np.int)



def train():

    model=test3()
    model.compile(optimizer='SGD',#keras.optimizers.Adam(lr=1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy']
                  )

    model.fit(train_x, train_y,
              batch_size=32,
              validation_data=(val_x, val_y),
              validation_steps=10,
              callbacks=[keras.callbacks.EarlyStopping(patience=10)],
              epochs=80 ,
              verbose=2,
              )
    return model
best=0
for x in range(100):

    model=train()
    result=model.evaluate(test_x,test_y,batch_size=1)
    if result[1]>best:
        best=result[1]
        print("best",best)
        os.remove(default_model_save_root+"best.h5")
        model.save(default_model_save_root+"best.h5")

