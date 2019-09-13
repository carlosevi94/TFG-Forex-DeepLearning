import argparse
import json

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

parse = argparse.ArgumentParser()
parse.add_argument('--currency', type=str)

args = parse.parse_args()
currency = args.currency.lower()

NOMBRE_TABLA = f"realtime_{currency}"


# =============================================================================
# Methods
# =============================================================================
def get_config(path):
    config = None
    with open(path) as json_file:
        config = json.load(json_file)
    return config['kafka'], config['mysql']


def guardar_mysql(row):
    import mysql.connector as sql
    import datetime

    # print(row)
    values = [row[1], row[0]]
    db_connection = sql.connect(host=mysql_config['host'], database=mysql_config["database"], user=mysql_config["user"], port=mysql_config['port'],
                                password=mysql_config["password"])
    db_cursor = db_connection.cursor()
    sql = "INSERT INTO " + NOMBRE_TABLA + " (id, value) VALUES (%s, %s)"
    db_cursor.execute(sql, values)
    db_connection.commit()
    print(f"{datetime.datetime.now()} - Success")


# =============================================================================
# Spark Session
# =============================================================================
spark = SparkSession.builder.appName('SparkStreamingKafka').getOrCreate()

# =============================================================================
# Config 
# =============================================================================
kafka_config, mysql_config = get_config("spark_consumer_config.json")

topic = f"forex_{currency}"
# =============================================================================
# Read from Kafka
# =============================================================================
print(kafka_config["brokers"])
df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_config["brokers"]) \
    .option("auto.offset.reset", "latest") \
    .option("value.deserializer", "StringDeserializer") \
    .option("subscribe", topic) \
    .load()

# =============================================================================
#                   ETL
# El formato del stream de kafka es: EUR/USD - 1970-01-19 00:56:15 - 1.128109
# =============================================================================
df = df.selectExpr("CAST(value AS STRING)") \
    .withColumn('DATE', to_timestamp(split(col('value'), ' - ')[1], 'yyyy-MM-dd HH:mm')) \
    .withColumn('VALUE', split(col('value'), ' - ')[2].cast(DoubleType()))

# =============================================================================
# Guardar en MySql cada dato que entra
# =============================================================================
query = df.writeStream.foreach(guardar_mysql).start()

# =============================================================================
# Guardar en csv cada 5 minutos
# =============================================================================
# query2 = df.writeStream \
#             .format('csv')\
#             .option('path','/home/carlos/Escritorio/csv_kafka')\
#             .option("checkpointLocation", "/home/carlos/Escritorio/csv_kafka_temp")\
#             .trigger(processingTime='5 minutes') \
#             .option('truncate', False) \
#             .start()

query.awaitTermination()
