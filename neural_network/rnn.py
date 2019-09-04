#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 13:20:35 2019

@author: carlos
"""

from warnings import filterwarnings
filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import missingno as msno
import scipy as scp
import seaborn as sns
from tqdm import tqdm
import datetime 

import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow.keras import layers

from tensorflow.keras.callbacks import EarlyStopping



from lib.utils import *



# =============================================================================
# Data
# =============================================================================
dataset = pd.read_csv('../data/DAT_ASCII_EURUSD_M1_2018.csv',
                      sep=";",
                      names=['date','open','high','low','close','volume'], 
                      index_col='date', 
                      parse_dates=['date'])
dataset.head()


# =============================================================================
# Train & Test
# =============================================================================

sc = MinMaxScaler(feature_range=(0,1))
AUX1 =  sc.fit_transform(np.array(dataset["close"]).reshape(-1,1))
dataset["close_escalado"] = AUX1

#He usado 20 dias para el train
train = dataset["2018-07-21":"2018-07-23"]['close_escalado']
test = dataset["2018-07-24"]['close_escalado']


(train.shape,test.shape)

train.tail()
test.head()


grafico_train_test(train,test)


# Pasar a numpy
train = np.array(train)
test = np.array(test)



train.shape





# =============================================================================
# Variables para longitudes secuenciales
# =============================================================================
predictor_range=180
prediction_range=45



# =============================================================================
# Secuencias
# =============================================================================
X_train = []
y_train = []
for i in range(predictor_range,(train.shape[0]-prediction_range)):
    X_train.append(train[i-predictor_range:i])
    y_train.append(train[i:i+prediction_range])
X_train, y_train = np.array(X_train), np.array(y_train)


X_train = np.reshape(X_train, (-1,1,predictor_range))
#y_train = np.reshape(y_train, (-1,1,prediction_range))
y_train = np.reshape(y_train, (-1,prediction_range))

(X_train.shape,y_train.shape)


# =============================================================================
# Evaluacion modelo
# =============================================================================
# Cojo la ultima secuencia del train, para predecir la siguiente
cadena_ultima_X_Y = list(X_train[-1].reshape(-1)) + list(y_train[-1].reshape(-1))
cadena_ultima_X_Y = np.array(cadena_ultima_X_Y[-predictor_range:]).reshape(1,1,predictor_range)
eval_y = test[:prediction_range].reshape(1,1,prediction_range)


# =============================================================================
# Modelo DL 
# =============================================================================
def get_model():
    regressor = tf.keras.Sequential()
    
    regressor.add(layers.Bidirectional(layers.LSTM(units=128,recurrent_dropout=0.175, return_sequences=True), input_shape=(X_train.shape[1],X_train.shape[2])))
#    regressor.add(Dropout(0.15))
#    regressor.add(SpatialDropout1D(0.15))
#    regressor.add(BatchNormalization())
    
#    regressor.add(LSTM(units=128,dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
#    regressor.add(BatchNormalization())
    
    regressor.add(layers.Bidirectional(layers.LSTM(units=128, return_sequences=True)))
#    regressor.add(SpatialDropout1D(0.3))
#    regressor.add(BatchNormalization())
    
#    regressor.add(LSTM(units=128 ,return_sequences=True))
#    regressor.add(SpatialDropout1D(0.15))
#    regressor.add(BatchNormalization())
    
    regressor.add(layers.TimeDistributed(layers.Dense(prediction_range)))
    return regressor

model = get_model()
model.summary()
model.compile(optimizer=tf.train.AdamOptimizer(),loss='mean_squared_error')
#model.compile(optimizer='adam',loss='mse')



hypams = {
        'epochs':150,
        'batch_size':32,
        'verbose': 1,
        'early_stopping':5,
        'min_delta':0.000001
    }
callbacks = [EarlyStopping(monitor='loss', min_delta=hypams['min_delta'],patience=hypams['early_stopping'],verbose=0,mode='min',restore_best_weights=True)]
history = model.fit(X_train,
                        y_train,
                        epochs=hypams['epochs'],
                        verbose=hypams['verbose'],
                        callbacks=callbacks, 
                        batch_size=hypams['batch_size'])#,
#                         validation_data=(X_test,y_test))
    
evaluacion_secuencial_2(model,cadena_ultima_X_Y,eval_y)




# =============================================================================
# Otros
# =============================================================================
grafico(history)    
evaluacion_secuencial()



#eval_X = X_train[-1].reshape(1,X_train[-1].shape[0],X_train[-1].shape[1]) 
#eval_y = y_train[-1].reshape(1,y_train[-1].shape[0],y_train[-1].shape[1])    
evaluacion_secuencial_2(cadena_ultima_X_Y,eval_y)

