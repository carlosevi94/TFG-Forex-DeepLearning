import json
from warnings import filterwarnings

import mysql.connector as sql
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

from lib.utils import *
import time
import numpy as np
import pandas as pd
import datetime

filterwarnings('ignore')


def update_tables(currency):
    with open('./config.json') as json_file:
        file = json.load(json_file)
        config = file['mysql']

    db_connection = sql.connect(host=config['host'], database=config['database'], user=config['user'],
                                password=config['password'], port=9306)

    db_cursor = db_connection.cursor()

    db_cursor.execute('SELECT * FROM realtime_%s order by id desc LIMIT 15' % currency)
    for n, row in enumerate(list(db_cursor.fetchall())):
        id, value = row
        db_cursor.execute('INSERT INTO historic_%s (id, value) VALUES ("%s", %s) ON DUPLICATE KEY UPDATE id=id' % (currency, id, value))
        if n != 0:
            db_cursor.execute('DELETE FROM realtime_%s where id = %s' % (currency, id))

    db_connection.commit()
    db_connection.close()


def main(currency):
    # update_tables(currency)
    time.sleep(3)
    with open('config.json') as json_file:
        file = json.load(json_file)
        config = file['mysql']
    # =============================================================================
    # Data
    # =============================================================================
    db_connection = sql.connect(host=config['host'], database=config['database'], user=config['user'],
                                password=config['password'], port=9306)

    dataset = pd.read_sql_query('SELECT id, value from historic_%s order by id desc LIMIT 180' % currency,
                                db_connection, coerce_float=False)
    db_connection.close()

    # =============================================================================
    # Train & Test
    # =============================================================================

    sc = MinMaxScaler(feature_range=(0, 1))
    AUX1 = sc.fit_transform(np.array(dataset["value"]).reshape(-1, 1))
    dataset["close_escalado"] = AUX1

    train = dataset['close_escalado']

    # Pasar a numpy
    train = np.array(train)

    # =============================================================================
    # Variables para longitudes secuenciales
    # =============================================================================
    predictor_range = 180
    prediction_range = 45

    # =============================================================================
    # Secuencias
    # =============================================================================
    X_train = train
    y_train = train[-45:]

    X_train = np.reshape(X_train, (-1, 1, predictor_range))
    y_train = np.reshape(y_train, (-1, prediction_range))


    # =============================================================================
    # Evaluacion modelo
    # =============================================================================
    # Cojo la ultima secuencia del train, para predecir la siguiente
    cadena_ultima_X_Y = list(X_train[-1].reshape(-1)) + list(y_train[-1].reshape(-1))
    cadena_ultima_X_Y = np.array(cadena_ultima_X_Y[-predictor_range:]).reshape(1, 1, predictor_range)


    # =============================================================================
    # Modelo DL
    # =============================================================================
    model = tf.keras.models.load_model('RNN_%s.h5' % currency)

    model.compile(optimizer=tf.train.AdamOptimizer(), loss='mean_squared_error')

    hypams = {
            'epochs': 1,
            'batch_size': 32,
            'verbose': 1,
            'early_stopping': 5,
            'min_delta': 0.000001
        }
    callbacks = [EarlyStopping(monitor='loss', min_delta=hypams['min_delta'],patience=hypams['early_stopping'],verbose=0,mode='min',restore_best_weights=True)]
    model.fit(X_train,
              y_train,
              epochs=hypams['epochs'],
              verbose=hypams['verbose'],
              callbacks=callbacks,
              batch_size=hypams['batch_size'])


    # GENERAR PREDICCION

    prediction = sc.inverse_transform(model.predict(cadena_ultima_X_Y))[0]

    # GUARDAR PREDICCION EN LA BASE DE DATOS
    db_connection = sql.connect(host=config['host'], database=config['database'], user=config['user'],
                                password=config['password'], port=9306)
    db_cursor = db_connection.cursor()

    first_date = dataset.iloc[0]['id']

    first_value = dataset.iloc[0]['value']

    date_range = np.array([first_date + datetime.timedelta(minutes=i) for i in range(1, 46)])

    db_cursor.execute(
        'INSERT INTO prediction_%s_rnn (id, value) VALUES ("%s", %s) ON DUPLICATE KEY UPDATE value=%s' % (
            currency, first_date, first_value, first_value))

    for n in range(len(prediction)):
        db_cursor.execute(
            'INSERT INTO prediction_%s_rnn (id, value) VALUES ("%s", %s) ON DUPLICATE KEY UPDATE value=%s' % (
                currency, date_range[n], prediction[n], prediction[n]))

    db_connection.commit()
    db_connection.close()

    model.save("RNN_%s.h5" % currency)


