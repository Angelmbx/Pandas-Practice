import pandas as pd
import numpy as np

df = pd.read_csv("dataset_1.csv", index_col=0)

print(df)
print("---- Describe -----")
print(df.describe())

'''
--- Describe -----
                Edad       Ingresos         Altura          Hijos   Muestra los valores numéricos
count  100000.000000  100000.000000  100000.000000  100000.000000   Total de 100.000 registros de cada columna
mean       53.931110   62590.579040       1.749618       1.941560   Media de 53 años, 62590 ingreos, 1'74 de altura y 1'94 hijos
std        40.980893   36529.619354       0.144122       2.353844   
min       -10.000000   -2000.000000       1.500000      -5.000000   Valores mínimos. Que la edad por ejemplo de negativo, demuestra que hay valores anómalos o erróneos
25%        23.000000   37658.750000       1.630000       0.000000
50%        51.000000   59832.000000       1.750000       2.000000
75%        78.000000   82207.250000       1.870000       4.000000
max       200.000000  199994.000000       2.000000       5.000000   Valores máximos. Que alguien tenga 200 años debería ser un valor anómalo
'''

print("---- Info -----")
print(df.info())

'''
 #   Column           Non-Null Count   Dtype  
---  ------           --------------   -----  
 0   Edad             100000 non-null  int64      100.000 registros no nulos y de tipo int64
 1   Género           90000 non-null   object     90.000 no nulos de tipo objeto
 2   Ingresos         100000 non-null  int64      etc...  
 3   Altura           100000 non-null  float64
 4   Ciudad           90000 non-null   object 
 5   Nivel_Educación  77563 non-null   object 
 6   Hijos            100000 non-null  int64 
 dtypes: float64(1), int64(3), object(3)          Tipos de datos en la tabla y cuántas veces se repiten
memory usage: 6.1+ MB                             Cuanto ocupan los datos
'''

# Para analizar las columnas categóricas (no numéricas), convertiremos cada una de ellas en una lista y posteriormente a un set
set_gen = set(df["Género"].to_list())
set_edu = set(df["Nivel_Educación"].to_list())
set_ciu = set(df["Ciudad"].to_list())

# Muestra todos los valores posibles que toman a lo largo de la tabla
print(set_gen) # {'F', nan, 'M'} 
print(set_edu) # {'Master', nan, 'no education', 'Bachelor', 'PhD', 'pHd', 'mastre', 'Bachelors'}
print(set_ciu) # {'New York', nan, 'Houston', 'Phoenix', 'Chicago', 'Los Angeles'}
# Más adelante veremos que estas categorias pueden ser convertidas a número (labeling coding)


## 1 Corregir algunos de los valores negativos, columna por columna
df["Edad"] = df["Edad"].apply(lambda x: np.nan if x < 0 else x) # Para cada valor negativo de edad, aplicaremos una lambda que sustiurá cada uno de ellos por un valor NaN, si no es negativo se mantiene el valor original
df["Ingresos"] = df["Ingresos"].apply(lambda x: np.nan if x<0 else x)
df["Hijos"] = df["Hijos"].apply(lambda x: np.nan if x<0 else x)

## 2 Imputar valores faltantes
for column in ["Edad", "Ingresos", "Hijos"]:
    median_value = df[column].median() # Para cada una de esas columnas, obtenemos su valor medio
    df[column].fillna(median_value, inplace=True) # Para cada columna, allá donde haya un valor NaN lo sustiuremos por su valor medio 
    # el inplace = True, provoca que se modifique el objeto original. Si está a False, crea una copia con los valores modificados
    # Cuando inplace = True no es necesario guardarlo en una nueva variable. Si fuera false es obligatorio guardarlo en una variable para recibir la copia del objeto: variable = df.metodo(inplace=False)

for column in["Género", "Ciudad"]:
    mode_value = df[column].mode()[0] # obtenemos el valor que mas se repite para la columna (la moda). Posición 0 porque primero te devuelve el valor, y en segunda posición cuántas veces se repite
    df[column].fillna(median_value, inplace=True) # sustuimos los valores NaN por la moda

## 3 Mapeo de datos -> Corrección de erratas
education_mapping = {
    "Bachelors" : "Bachelor",
    "mastre" : "Master",
    "pHd" : "PhD",
    "no education" : "NE"
}

df["Nivel_Educación"].replace(education_mapping, inplace=True) # Corrige los errores en esa columna
df["Nivel_Educación"].fillna("NE", inplace=True) # Sustituye los nulos por "NE"

## 4 Casteo de tipos -> Convertimos los datos de cada columna a los tipos que realmente queremos 
df["Edad"] = df["Edad"].astype(int)
df["Hijos"] = df["Hijos"].astype(int)
df["Ingresos"] = df["Ingresos"].astype(float)
df["Altura"] = df["Altura"].astype(float)

print(df.info())

'''
Ahora ya no muestra valores no nulos

 #   Column           Non-Null Count   Dtype  
---  ------           --------------   -----  
 0   Edad             100000 non-null  int64  
 1   Género           100000 non-null  object 
 2   Ingresos         100000 non-null  float64
 3   Altura           100000 non-null  float64
 4   Ciudad           100000 non-null  object 
 5   Nivel_Educación  100000 non-null  object 
 6   Hijos            100000 non-null  int64  
dtypes: float64(2), int64(2), object(3)
memory usage: 6.1+ MB
'''

