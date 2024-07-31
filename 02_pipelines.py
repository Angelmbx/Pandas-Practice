import pandas as pd
import numpy as np

## Funciones a las que les das un dataframe y te devuelve un dataframe corregido

df = pd.DataFrame("datase_1.csv", index_col=0)

# en 01_data_processing.py corregimos los valores negativos columna por columna. En esta ocasión implementaremos una función que sirva para cualquier columna

def remove_negative_values(df, column):
    df[column] = df[column].apply(lambda x: np.nan if x<0 else x)

'''
Z-score:
es una medida estadística que describe la posición de un valor individual dentro de un conjunto de datos en relación con la media del conjunto y la desviación estándar.
Específicamente, el Z-score indica cuántas desviaciones estándar se encuentra un valor por encima o por debajo de la media.

Cálculo del Z-score
El Z-score de un valor x:

Z= (x − μ) / σ

Donde:
x es el valor individual.
μ es la media del conjunto de datos.
σ es la desviación estándar del conjunto de datos.

Z-score = 0: El valor x es igual a la media.
Z-score > 0: El valor x está por encima de la media.
Z-score < 0: El valor x está por debajo de la media.
Z-score > 2 o Z-score < -2: El valor x puede considerarse un outlier, ya que está más de 2 desviaciones estándar por encima o por debajo de la media. El umbral de 2 es común, pero puede variar según el contexto.
'''
# función para datos anómalos (p.ej. Edad = 200)
def remove_outliers_with_zscore(df, column, threshold = 2): # threshold: umbral que define cuántas desviaciones estándar por encima o por debajo de la media se considera un valor atípico. En este caso cualquier valor a más de 2 desviaciones estándar de la media -> outlier
    column_mean = df[column].mean() # lo primero que necesitamos es el promedio de la columna
    column_std = df[column].std() # y la desviación estandar
    df[column] = df[column].mask(((df[column] - column_mean) / column_std).abs() > threshold, column_mean)
   # .abs() incluye tanto los valores que esten a +2 desviaciones de la media tanto por encima como por debajo. Sin él solo se tiene en cuenta los que estén por encima.
   # el column_mean reemplaza los valores identificados como outliers con la media de la columna.    
    return df