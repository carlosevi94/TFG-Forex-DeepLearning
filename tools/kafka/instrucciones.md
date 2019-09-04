


## Descripcion
El docker utilizado es el de [confluent inc](https://hub.docker.com/r/confluentinc/cp-kafka/). Aquí hay unas [instrucciones orientativas](https://docs.confluent.io/3.2.1/installation/docker/docs/quickstart.html).

El docker compose se ha modificado respecto al original para que funcione perfectamente con la versión 2.4 de Spark.


## Comandos

- __Levantar docker__
```bash
docker-compose up -d
```

- __Parar docker__
```bash
docker-compose stop
```

- __Crear TOPIC__
```bash
docker-compose exec kafka kafka-topics --create --topic __NAME__ --partitions 1 --replication-factor 1 --if-not-exists --zookeeper __ZOOKEEPER_IP__:2181
```
Sustituir `__NAME__` por el nombre que quieras  
Sustituir `__ZOOKEEPER_IP__` por la ip del container de zookeeper

Para este sistema
```bash
docker-compose exec kafka kafka-topics --create --topic forex_eurusd --partitions 1 --replication-factor 1 --if-not-exists --zookeeper 172.20.0.2:2181
```

- __Verificar topic creado__
```bash

docker-compose exec kafka  \
  kafka-topics --describe --topic forex_eurusd --zookeeper 172.20.0.3:2181
```

- __Consumer de Kafka__  
Para este sistema
```bash
docker-compose exec kafka  kafka-console-consumer --bootstrap-server 172.20.0.3:9092 --topic forex_eurusd --new-consumer --from-beginning --max-messages 42

```
