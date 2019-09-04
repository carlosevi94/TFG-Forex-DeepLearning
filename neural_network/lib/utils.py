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

from sklearn.metrics import mean_squared_error



def grafico(history):
    plt.figure(figsize=(18,5))
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
    
def grafico_train_test(train,test,prediction=None,debug=False):
    plt.figure(figsize=(18,5))
    eje_x = [x for x in range(train.shape[0] + test.shape[0])]
    plt.plot(eje_x[:train.shape[0]],train,label='Train')
    plt.plot(eje_x[train.shape[0]:],test,label='Test',color="orange")
    plt.plot(eje_x[train.shape[0]-1:train.shape[0]+1],np.concatenate((train[-1],test[0]), axis=None),label=None,color="orange")

    if debug:
        plt.grid()

    try:
        plt.plot(eje_x[train.shape[0]:],prediction,label='Prediction',color="green")
        plt.plot(eje_x[train.shape[0]-1:train.shape[0]+1],np.concatenate((train[-1],prediction[0]), axis=None),label=None,color="green")

    except:
        pass    
    plt.legend(loc='upper left')    

def return_rmse(test,predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {}.".format(rmse))
    
def random_date(year,dias=15):
    import random
    start_date = datetime.date.today().replace(day=1, month=1,year=year).toordinal()
    end_date = datetime.date.today().toordinal()
    random_day = datetime.date.fromordinal(random.randint(start_date, end_date))
    random_day_2 = random_day + datetime.timedelta(dias)
    return str(random_day),str(random_day_2)    




def ventana_temporal(df,dias_train=15,dia_inicio=False):
    if dia_inicio:
        dia_fin =  str(datetime.datetime.strptime(dia_inicio,"%m-%d-%Y") + datetime.timedelta(dias_train))
    else:
        dia_inicio,dia_fin = random_date(2018,dias_train)
    print(dia_inicio)
    print(dia_fin)    
        
    train = df[dia_inicio:dia_fin]
    
    dia_test_fin = str(datetime.datetime.strptime(dia_fin,"%Y-%m-%d") + datetime.timedelta(1)).split(" ")[0]
    test = df[dia_test_fin]    
    
    return train,test




def evaluacion_secuencial():
    ultima_sec = train[-predictor_range:]
    ultima_sec = ultima_sec.reshape(1,1,predictor_range)

    primer_futuro_sec  = test[:prediction_range]
    primer_futuro_sec = primer_futuro_sec.reshape(1,1,prediction_range)



    predic_first_future = model.predict(ultima_sec)

    grafico_train_test(
        ultima_sec.reshape(-1),
        primer_futuro_sec.reshape(-1))

    grafico_train_test(
        ultima_sec.reshape(-1),
        primer_futuro_sec.reshape(-1),
        predic_first_future.reshape(-1)
    )

    return_rmse(primer_futuro_sec.reshape(-1),predic_first_future.reshape(-1))


def evaluacion_secuencial_2(model,train,test,debug=False):
    print(train.shape)
    print(test.shape)

    predic_first_future = model.predict(train)

    grafico_train_test(
        train.reshape(-1),
        test.reshape(-1),
        debug=debug)

    grafico_train_test(
        train.reshape(-1),
        test.reshape(-1),
        predic_first_future.reshape(-1),
        debug=debug
    )

    return_rmse(test.reshape(-1),predic_first_future.reshape(-1))
    return predic_first_future

def evaluacion_secuencial_3(model,train,test,sc,debug=False):
    print(train.shape)
    print(test.shape)

    predic_first_future = model.predict(train)
    #predic_first_future = sc.inverse_transform(predic_first_future)
    print("PREDICCION {}".format(predic_first_future.tolist()))
#    print(predic_first_future)
    test = sc.inverse_transform(test)

    grafico_train_test(
        train.reshape(-1),
        test.reshape(-1),
        debug=debug)

    grafico_train_test(
        train.reshape(-1),
        test.reshape(-1),
        predic_first_future.reshape(-1),
        debug=debug
    )

    return_rmse(test.reshape(-1),predic_first_future.reshape(-1))
    return predic_first_future

