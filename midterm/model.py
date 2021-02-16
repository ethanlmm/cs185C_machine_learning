from tensorflow import  keras
import tensorflow as tf


def logit(x):
    return keras.backend.log(1 / (1 + keras.backend.exp(-x)))
def binary(x):
    return 0 if(x<0) else 1
def test1():
    inputs=keras.layers.Input(shape=(10,))
    dnn_1=keras.layers.Dense(128,activation='relu')(inputs)
    dnn_1 = keras.layers.Dropout(0.5)(dnn_1)

    dnn_2 = keras.layers.Dense(64, activation='relu')(dnn_1)
    dnn_2 = keras.layers.Dropout(0.5)(dnn_2)

    dnn_3 = keras.layers.Dense(32, activation='relu')(dnn_2)
    dnn_3 = keras.layers.Dropout(0.5)(dnn_3)

    dnn_4 = keras.layers.Dense(16, activation='relu')(dnn_3)
    dnn_4 = keras.layers.Dropout(0.5)(dnn_4)

    merge=keras.layers.Concatenate(axis=1)([dnn_1,dnn_2,dnn_3,dnn_4])
    result=keras.layers.Dense(10,activation='relu')(merge)
    result=keras.layers.Dense(2,activation='softmax')(result)
    model=keras.Model(inputs=inputs,outputs=result)
    model.summary()
    return model

def test():
    inputs=keras.layers.Input(shape=(9,))
    dnn_s=keras.layers.Dense(32,activation='relu')(inputs)
    dnn_s = keras.layers.Dropout(0.5)(dnn_s)
    dnn_s = keras.layers.Dense(32, activation='relu')(dnn_s)
    dnn_s =keras.layers.Dropout(0.5)(dnn_s)
    dnn_s = keras.layers.Dense(32, activation='relu')(dnn_s)
    dnn_s = keras.layers.Dropout(0.5)(dnn_s)
    dnn_s = keras.layers.Dense(32, activation='relu')(dnn_s)

    dnn_m = keras.layers.Dense(64,activation='relu')(inputs)
    dnn_m = keras.layers.Dense(64, activation='relu')(dnn_m)
    dnn_m = keras.layers.Dropout(0.5)(dnn_m)
    dnn_m = keras.layers.Dense(64, activation='relu')(dnn_m)
    dnn_m = keras.layers.Dropout(0.5)(dnn_m)
    dnn_m = keras.layers.Dense(64, activation='relu')(dnn_m)
    dnn_m = keras.layers.Dropout(0.5)(dnn_m)
    dnn_m = keras.layers.Dense(64, activation='relu')(dnn_m)
    dnn_m = keras.layers.Dropout(0.5)(dnn_m)
    dnn_m = keras.layers.Dense(64, activation='relu')(dnn_m)

    dnn_l = keras.layers.Dense(128,activation='relu')(inputs)
    dnn_l = keras.layers.Dropout(0.5)(dnn_l)
    dnn_l = keras.layers.Dense(128, activation='relu')(dnn_l)
    dnn_l = keras.layers.Dropout(0.5)(dnn_l)
    dnn_l = keras.layers.Dense(128, activation='relu')(dnn_l)
    dnn_l = keras.layers.Dropout(0.5)(dnn_l)
    dnn_l = keras.layers.Dense(128, activation='relu')(dnn_l)
    dnn_l = keras.layers.Dropout(0.5)(dnn_l)
    dnn_l = keras.layers.Dense(128, activation='relu')(dnn_l)
    dnn_l = keras.layers.Dropout(0.5)(dnn_l)
    dnn_l = keras.layers.Dense(128, activation='relu')(dnn_l)
    dnn_l = keras.layers.Dropout(0.5)(dnn_l)
    dnn_l = keras.layers.Dense(128, activation='relu')(dnn_l)
    dnn_l = keras.layers.Dropout(0.5)(dnn_l)
    dnn_l = keras.layers.Dense(128, activation='relu')(dnn_l)

    merge=keras.layers.Concatenate(axis=1)([dnn_s,dnn_m,dnn_l])
    result=keras.layers.Dense(64,activation='relu')(merge)
    result = keras.layers.Dense(32, activation='relu')(result)
    result = keras.layers.Dense(16, activation='relu')(result)
    result=keras.layers.Dense(2,activation='softmax')(result)
    model=keras.Model(inputs=inputs,outputs=result)
    model.summary()
    return model

def test2():
    model = tf.keras.Sequential([

        tf.keras.layers.Dense(64, input_shape=(10,),activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax)
    ])
    return model
def test3():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, input_shape=(10,), activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax)
    ])
    return model
