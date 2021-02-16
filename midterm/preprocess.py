
from sklearn import preprocessing
from datetime import datetime
import numpy as np
import os
import re
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

def load_data(root,type):
    x = ls(root, type)
    return np.loadtxt(x[0], delimiter=',')

def save_data(root,x,y):

    target = y
    train = x
    train_count = int(train.shape[0] * 0.8)
    val_count = int(train.shape[0] * 0.1)

    train_x = train[:train_count]
    train_y = target[:train_count]

    val_x = train[train_count:train_count + val_count]
    val_y = target[train_count:train_count + val_count]

    test_x = train[train_count + val_count:]
    test_y = target[train_count + val_count:]


    os.mkdir(root)
    np.savez(root + "train", inputs=train_x, targets=train_y)
    np.savez(root + "val", inputs=val_x, targets=val_y)
    np.savez(root + "test", inputs=test_x, targets=test_y)





def balanced_preprocess(root):
    raw= load_data(root,csv)
    np.random.shuffle(raw)
    balanced = []
    nOfOne = 0
    nOfZero = 0
    for x in raw:
        if x[-1] == 1:
            balanced.append(x)
            nOfOne += 1
    for x in raw:
        if nOfOne < nOfZero:
            break
        if x[-1] == 0:
            balanced.append(x)
            nOfZero += 1



    data =np.array(balanced)
    np.random.shuffle(data)
    np.random.shuffle(data)
    path = root + "balance/" + datetime.now().strftime("%Y-%m-%d %H-%M%S")+"/"
    target = data[:, -1]
    onehot = []
    for x in target:
        if x==0:
            onehot.append([1,0])
        if x==1:
            onehot.append([0,1])
    scaled = preprocessing.scale(data[:,1:-1])
    xbar=[]
    for x in range(data.shape[0]):
        if x%100==0:
            y=np.sum(target[x-100:x])/100
            xbar.append(y)
            print(y)
    print(np.sum(xbar)/len(xbar))
    print(scaled.shape)
    print(np.sum(target)/target.shape[0])

    save_data(path,scaled,onehot)

balanced_preprocess(default_root)

