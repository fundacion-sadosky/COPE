cope_rf_industrial
==============================

Este repositorio contiene los scripts y análisis realizados durante la ejecución del proyecto COPE - “Sistema inteligente de medición de nivel y control de velocidad de bombeo para pozos petrolíferos".

Dentro del directorio notebooks puede encontrar todos los análisis desarrollados a lo largo del proyecto. A continuación se introduce un índice breve detallando la referencia del número inicial que acompaña al nombre de cada uno de ellos:

0: Código provisto por la empresa al comenzar el proyecto para leer datos generados.

1: Primeros análisis exploratorios y búsqueda de métodos básicos para eliminar la saturación en las señales registradas.

2: Desarrollo de métodos de estimación básicos para calcular la velocidad del sonido en el pozo en metros por segundo.

3: Búsqueda de optimización de los parámetros de los distintos métodos a partir de los datos disponibles (solo para los pozos 8 y 9).

4: Comparativa de métodos en base al resultado de la optimización, que permite obtener el método que mejor resultados ofrece. También se incluye en este punto un notebook en el que se proponen nuevos pulsos a utilizarse que podrían mejorar los resultados obtenidos.

5: Análisis exploratorios sobre señales para cálculo de la profundidad del pozo.

6: Adaptaciones en los métodos desarrollados inicialmente para realizar una estimación de la profundidad del pozo.

7: Optimización de los métodos de estimación de profundidad.

8: Análisis exploratorio sobre los datos disponibles de dinamometrías.

9: Exploración de distintas técnicas y enfoques para estimar variables de relevancia sobre el comportamiento del pozo. Se introduce una comparativa contra una línea base.

Nuevos datos
------------

En caso de contar con nuevos registros, se han desarrollado dos notebooks que permiten optimizar nuevamente los parámetros de los métodos desarrollados y obtener las predicciones de velocidad del sonido y de profundidad de los pozos. Los mismos son: notebooks/Optimizacion_Metodos.ipynb y notebooks/Obtener_Predicciones.ipynb



Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         and a short `_` delimited description, e.g.
    │                         `1_initial_data_exploration`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │
        └── methods        <- Scripts to use proposed methods.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
