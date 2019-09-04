### Paso 0: 
`sudo apt-get update`

### Paso 1: Instalar Java

`sudo apt-get install default-jdk`

### Paso 2: Instalar Scala

`sudo apt-get install scala`

### Paso 3: Instalar Spark

1. Ir a `http://spark.apache.org/downloads.html`
2. Descargarlo (es el paso 3 en la web)
3. Descomprimirlo: `sudo tar xvf spark-blabla.tgz /usr/local/`
4. `sudo mv /usr/local/spark_VERSION /usr/local/spark`
5. Añadimos variables de entorno:  
5.1. `nano ~/.bashrc` o `nano ~/.zshrc`  
5.2. Añadimos

```bash
export SPARK_HOME=/usr/local/spark
export PYSPARK_PYTHON=python3

export PATH=$SPARK_HOME/bin:$PATH
```

6. `source ~/.bashrc` o `source ~/.zshrc`


## Paso 4: Instalar Jars necesarios para la comunicación con Kafka

Ir al `$SPARK_HOME/jars` y pegar los 3 jars que hay en la carpeta `jars_to_install`


### Listo!
Abre una consola de python3 y escribe `import pyspark` para comprobarlo


### Fallos Comunes:
Es bastante comun que cuando escribas import pyspark falle porque no encuentra la libreria py4j.
Hay 2 soluciones, cualquiera de las 2 opciones es igual de valida.

1. `pip3 install py4j`
2. Añadir nuevas variables de entorno   
2.1. `ls $SPARK_HOME/python/lib/ | ps py4j`  
2.2. Copia la salida de ese comando e introducela en SALIDA_LS de la linea siguiente.  
`export PYTHONPATH=$SPARK_HOME/python/lib/py4j-{SALIDA_2.1}-src.zip:$PYTHONPATH`  
2.3. Ahora, añadimos estas variables de entorno en `nano ~/.bashrc` o `nano ~/.zshrc`, y luego repites el paso 6   

```bash
export PYTHONPATH=$SPARK_HOME/python/:$PYTHONPATH
export PYTHONPATH=$SPARK_HOME/python/lib/py4j-{SALIDA_2.1}-src.zip:$PYTHONPATH
```




#### Otros:  
Esta variable supuestamente es obligatoria. NO obstante, he instalado varias veces Spark sin esta libreria, y no falla en ningun sitio. ahí la dejo por si acaso.
```bash
export PYTHONPATH=$SPARK_HOME/python:$PYTHONPATH
```



