import pandas as pd
import sqlalchemy
from sqlalchemy.types import TIMESTAMP, Float
import json
import os

with open("./config.json") as json_file:
    file = json.load(json_file)
    config = file["mysql"]

HOST = config['host']    

DB = config['database']

PORT = config['port']

USER = config['user']

PASSWORD = config['password']

FILENAME_EURUSD = config['eurusd_file']

URL = config['eurusd_url']

engine = sqlalchemy.create_engine('mysql+mysqlconnector://' + USER + ':' + PASSWORD + '@' + HOST + ':' + PORT + '/' + DB, echo=False)

df_eur = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + "/" + FILENAME_EURUSD)

df_eur['date'] = pd.to_datetime(df_eur['date'])

df = pd.read_csv(URL, names=['date', 'open', 'high', 'low', 'close'], skiprows=1)

df['date'] = pd.to_datetime(df['date'])

df_eur = pd.concat([df_eur, df], ignore_index=True)

if not df_eur[df_eur.duplicated()].empty:
    df_eur = df_eur.drop_duplicates(df['date'])

subdf = df_eur[['date', 'close']]

subdf = subdf.set_index('date', drop=True)

subdf = subdf.rename(columns={"date": "id", "close": "value"})

try:
    subdf.to_sql(name='historic_eur_usd', con=engine, if_exists='append', index_label='id', dtype={'date': TIMESTAMP, 'value': Float(10)})
except:
    print("Ya existen datos en la BD")

engine.dispose()

df_eur.to_csv(os.path.dirname(os.path.realpath(__file__)) + "/" + FILENAME_EURUSD, index=False)