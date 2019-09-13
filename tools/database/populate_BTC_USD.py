import pandas as pd
import sqlalchemy
from sqlalchemy.types import TIMESTAMP, NUMERIC
import json

with open("./config.json") as json_file:
    file = json.load(json_file)
    config = file["mysql"]

HOST = config['host']    

DB = config['database']

PORT = config['port']

USER = config['user']

PASSWORD = config['password']

FILENAME_BTC_USD = config['btcusd_file']

URL = config['btcusd_url']

engine = sqlalchemy.create_engine('mysql+mysqlconnector://' + USER + ':' + PASSWORD + '@' + HOST + ':' + PORT + '/' + DB, echo=False)

df_btc = pd.read_csv(URL, skiprows=1)
df_btc['Date'] = pd.to_datetime(df_btc['Date'])


value_serie = df_btc[['Date', 'Close']]
value_serie = value_serie.set_index('Date', drop=True)
value_serie = value_serie.rename(columns={"Date": "id", "Close": "value"})

try:
	value_serie.to_sql(name='historic_btc_usd', con=engine, if_exists='append', index_label='id', dtype={'date': TIMESTAMP, 'value': NUMERIC})
except:
	print("Ya existen datos en la BD")

engine.dispose()

df_btc.to_csv(FILENAME_BTC_USD)