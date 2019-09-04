# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


import datetime 



def grafico_retornos_train_test(train,test,pred,debug=False):
    plt.figure(figsize=(18,5))
    
    plt.plot(train,label='Train')
    plt.plot(test,label='Test',color="orange")
    

    aux_last_x_1 = train[-1]
    aux_last_x_2 = train.index[-1]
    aux_first_x_1 = test[0]
    aux_first_x_2 = test.index[0]
    serie_conectora_1 = pd.Series([aux_last_x_1,aux_first_x_1],index=[aux_last_x_2,aux_first_x_2])
    
    plt.plot(serie_conectora_1,label=None,color="orange")

    
    plt.plot(pred,color="green",label="Prediction")
    aux_first_x_1 = pred[0]
    aux_first_x_2 = pred.index[0]
    serie_conectora_1 = pd.Series([aux_last_x_1,aux_first_x_1],index=[aux_last_x_2,aux_first_x_2])
    
    plt.plot(serie_conectora_1,label=None,color="green")
    
    if debug:
        plt.grid()

    plt.legend(loc='upper left')    