import pandas as pd
import numpy as np

## Funciones a las que les das un dataframe y te devuelve un dataframe corregido

# 1) En 01_data_processing.py corregimos los valores negativos columna por columna. En esta ocasión implementaremos una función que sirva para cualquier columna

def remove_negative_values(df, column):
    df[column] = df[column].apply(lambda x: np.nan if x<0 else x)
    return df

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
# 2) Función para datos anómalos (p.ej. Edad = 200)
def remove_outliers_with_zscore(df, column, threshold = 2): # threshold: umbral que define cuántas desviaciones estándar por encima o por debajo de la media se considera un valor atípico. En este caso cualquier valor a más de 2 desviaciones estándar de la media -> outlier
    column_mean = df[column].mean() # lo primero que necesitamos es el promedio de la columna
    column_std = df[column].std() # y la desviación estandar
    df[column] = df[column].mask(((df[column] - column_mean) / column_std).abs() > threshold, column_mean)
   # .abs() incluye tanto los valores que esten a +2 desviaciones de la media tanto por encima como por debajo. Sin él solo se tiene en cuenta los que estén por encima.
   # el column_mean reemplaza los valores identificados como outliers con la media de la columna.   
    return df

# 3) Función para realizar mapeos de datos
def map_column_values(df, column, mapping_dict): 
    # mapping_dict: Un diccionario que contiene los mapeos de los valores originales a los nuevos valores. Las claves son los valores originales y los valores del diccionario son los valores nuevos.
    df[column] = df[column].apply(lambda value: mapping_dict.get(value, value)) 
    return df
# a cada valor de la columna le aplicamos una lambda.
# mapping_dict.get()-> retorna un valor dada su clave. 
# mapping_dict.get(value.lower().strip(), np.nan)-> Busca el valor convertido en minúsculas y sin espacios en el diccionario de mapeo. Si encuentra una coincidencia, devuelve el valor mapeado (corregido); si el valor no está en el diccionario, devuelve NaN.
# if value is not np.nan else np.nan: Realiza lo anterior cuando el valor original no es NaN. Si fuese NaN, se mantiene.

# 4) Función para rellenar todos aquellos datos NaN que haya tras nuestro tratamiento de datos
def fill_na_in_column(df, column, fill_value):
    df.fillna({column: fill_value}, inplace= True)
    return df

def preprocess_data(df):
    education_mapping = {
        "Bachelors" : "Bachelor",
        "mastre" : "Master",
        "pHd" : "PhD",
        "no education" : "NE"
    }

    gender_mapping = {
        "m" : "M",
        "f" : "F"
    }

    # Aquí devolvemos nuestro pipeline. Para ello crearemos una Tupla con todas las instrucciones que queremos que nuestro DataFrame siga
    return (
        df.pipe(remove_negative_values, "Edad")
        .pipe(remove_negative_values, "Ingresos") # Borra los valores negativos de Edad, Ingresos, Hijos
        .pipe(remove_negative_values, "Hijos")
        .pipe(remove_outliers_with_zscore, "Edad")
        .pipe(remove_outliers_with_zscore, "Ingresos") # Suprime los outliers de Edad, Ingresos, Altura, Hijos sustituyéndolos por la media de cada columna
        .pipe(remove_outliers_with_zscore, "Altura")
        .pipe(remove_outliers_with_zscore, "Hijos")
        .pipe(map_column_values, "Nivel_Educación", education_mapping) # Mapea los datos de Nivel_Educacion y Género por los datos de los diccionarios education_mapping y gender_mapping
        .pipe(map_column_values, "Género", gender_mapping)
        .pipe(fill_na_in_column, "Ciudad", "Desconocido")
        .pipe(fill_na_in_column, "Nivel_Educación", "Desconocido") # Sustituye los registros donde aparezca un NaN por "Desconocido"
        .pipe(fill_na_in_column, "Género", "Desconocido")
        .pipe(fill_na_in_column, "Edad", df["Edad"].median()) # En este caso elegimos que los registros NaN los sustituya por la mediana
        .pipe(fill_na_in_column, "Hijos", df["Hijos"].median())
        .pipe(fill_na_in_column, "Ingresos", df["Ingresos"].mean()) # Y en este por la media
        .pipe(fill_na_in_column, "Altura", df["Altura"].mean())
    )

df = pd.read_csv("dataset_1.csv", index_col=0)
print(df)
print("----------AFTER PROCESSING-----------")
df = preprocess_data(df)
print(df)

## En conclusión, este código es más robusto, fácil de escalar y mantener que el de 01_data_processing.py