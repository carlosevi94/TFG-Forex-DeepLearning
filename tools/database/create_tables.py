import mysql.connector as sql
import json

with open('./config.json') as json_file:
    file = json.load(json_file)
    config = file['mysql']

HOST = config['host']    

DB = config['database']

PORT = config['port']

USER = config['user']

PASSWORD = config['password']

db_connection = sql.connect(host=HOST, user=USER, password=PASSWORD, port=PORT)

db_cursor = db_connection.cursor()

db_cursor.execute('CREATE DATABASE IF NOT EXISTS %s' % DB)

db_connection.commit()

db_connection.close()

db_connection = sql.connect(host=HOST, database=DB, user=USER, password=PASSWORD, port=PORT)

db_cursor = db_connection.cursor()

db_cursor.execute('CREATE TABLE historic_eur_usd(id datetime not null primary key, value double)')
db_cursor.execute('CREATE TABLE historic_btc_usd(id datetime not null primary key, value double)')

db_cursor.execute('CREATE TABLE realtime_eur_usd(id datetime not null primary key, value double)')
db_cursor.execute('CREATE TABLE realtime_btc_usd(id datetime not null primary key, value double)')

db_cursor.execute('CREATE TABLE prediction_eur_usd_rnn(id datetime not null primary key, value double)')
db_cursor.execute('CREATE TABLE prediction_eur_usd_cnn(id datetime not null primary key, value double)')
db_cursor.execute('CREATE TABLE prediction_btc_usd_rnn(id datetime not null primary key, value double)')
db_cursor.execute('CREATE TABLE prediction_btc_usd_cnn(id datetime not null primary key, value double)')

db_connection.commit()

db_connection.close()
