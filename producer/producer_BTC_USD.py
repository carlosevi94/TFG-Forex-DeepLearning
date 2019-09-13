import sys
import time
from datetime import datetime
from string import Template

from confluent_kafka import Producer

from lib import scrapper
import schedule


BROKERS = 'localhost:9092'
TOPIC = 'forex_btc_usd'
URL = 'https://es.investing.com/crypto/bitcoin/btc-usd'
STOCK = 'BTC/USD'
RESPONSE_TEMPLATE = Template('$stock - $timestamp - $value')

conf = {
    'bootstrap.servers': BROKERS,
    'session.timeout.ms': 6000,
    'default.topic.config': {'auto.offset.reset': 'smallest'}
}
producer = Producer(**conf)


def delivery_callback(err, msg):
    if err:
        print('%% Message failed delivery: %s\n' % err)
    else:
        print('%% Message delivered to %s [%d]\n' % (msg.topic(), msg.partition()))


def prepare_value():
    value = scrapper.get_current_value(URL).replace(".","").replace(",",".")
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    res = RESPONSE_TEMPLATE.substitute(stock=STOCK, timestamp=timestamp, value=value)
    print(res)
    send_value(res)


def send_value(res, retry=0):
    try:
        producer.produce(TOPIC, res, callback=delivery_callback)
        producer.poll(0)
        sys.stderr.write('%% Waiting for %d deliveries\n' % len(producer))
        producer.flush()
    except Exception as e:
        if retry < 5:
            retry += 1
            send_value(res, retry)
        else:
            sys.stderr.write(e)


if __name__ == '__main__':
    schedule.every().minute.at(":00").do(prepare_value)
    while True:
        schedule.run_pending()
        time.sleep(1)
