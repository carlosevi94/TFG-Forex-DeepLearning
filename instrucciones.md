## Preparación del entorno

```bash 
sudo apt-get update
```

### Instalar Java

```bash 
sudo apt-get install openjdk-8-jdk
```

Añadir JAVA_HOME a las variables de entorno al archivo   `~/.bashrc`

```bash
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
````

### Instalar Scala

`sudo apt-get install scala`

### Instalar Spark

1. Descargar de `http://spark.apache.org/downloads.html`
2. Descomprimir: `sudo tar xzvf spark-VERSION.tgz`
3. Mover la carpeta descomprimida: `sudo mv PATH_SPARK_FOLDER/spark_VERSION /usr/local/spark`
4. Añadir variables de entorno al archivo `~/.bashrc`  

	```bash
	export SPARK_HOME=/usr/local/spark
	export PYSPARK_PYTHON=python3
	export PATH=$SPARK_HOME/bin:$PATH
	export PYTHONPATH=$SPARK_HOME/python:$PYTHONPATH
	```

5. `source ~/.bashrc`

#### Jar necesarios para Spark
Copiar el contenido de la carpeta `./tools/spark/jars` en  `/usr/local/spark/jars`


### Instalar Docker

```bash
sudo apt-get update
sudo apt install docker.io
```

```bash
sudo systemctl start docker
```

### Instalar Docker Compose

```bash
sudo curl -L "https://github.com/docker/compose/releases/download/1.23.1/docker-compose-\
	$(uname -s) -$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### Instalar dependencias
En la carpeta base del proyecto

```bash 
python3 -m venv venv
source venv/bin/activate
pip install -r tools/requirements.txt
```


## Arranque del sistema

Desde la carpeta base del proyecto

### Iniciar BD

```bash
cd tools/front_end
sudo docker-compose up -d
```
#### Crear tablas
```bash
cd ../database
python create_tables.py
```

#### Popular tablas
```bash
python populate_BTC_USD.py
python populate_EUR_USD.py
```

### Levantar Kafka
```bash
cd ../kafka
sudo docker-compose up -d

```
#### Obtener IP de los contenedos
Para obtener las ips del contenedor de zookeeper y kafka lanzar los siguientes comandos:

```bash 
sudo docker inspect kafka_cont
sudo docker inspect zookeeper_cont
```


#### Crear topics
Sustituir las los parametros ZOOKEPPER_IP y KAFKA_IP por sus respectivas IPsS

```bash 
sudo docker-compose exec kafka kafka-topics --create --topic forex_eur_usd --partitions 1 \
	--replication-factor 1 --if-not-exists --zookeeper _ZOOKEEPER_IP_:2181
sudo docker-compose exec kafka kafka-topics --create --topic forex_btc_usd --partitions 1 \
	--replication-factor 1 --if-not-exists --zookeeper _ZOOKEEPER_IP_:2181
```


### Conectar Spark
Desde la carpeta base del proyecto abrir dos terminales bash y lanzar en cada unos de los siguienes comandos (Precaución: Hay que activar el entorno virtual en cada consola nueva):

```bash
cd spark_scripts
python spark_consumer.py --currency eur_usd
```

```bash
cd spark_scripts
python spark_consumer.py --currency btc_usd
```

### Iniciar productores de datos
Desde la carpeta base del proyecto abrir dos terminales bash y lanzar en cada unos de los siguienes comandos (Precaución: Hay que activar el entorno virtual en cada consola nueva):

```bash
cd producer
python producer_EUR_USD.py
```

```bash
cd producer
python producer_BTC_USD.py
```

### Iniciar redes neuronales
Desde la carpeta base del proyecto abrir cuatro terminales bash y lanzar en cada unos de los siguienes comandos (Precaución: Hay que activar el entorno virtual en cada consola nueva):

```bash
cd neural_networks
python cnn.py --currency eur_usd
python cnn.py --currency btc_usd
python rnn.py --currency eur_usd
python rnn.py --currency btc_usd
```

### Desplegar front-end
Desde la carpeta base del proyecto abrir una terminal bash y lanzar los siguienes comandos (Precaución: Hay que activar el entorno virtual en cada consola nueva):

```bash
cd front_end/webapp
python app.py
```
Si todo ha ido correctamente el front-end debe estar accesible en la url `localhost:8050`

