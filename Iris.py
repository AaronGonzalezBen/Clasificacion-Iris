"""
CLASIFICADOR FLORES TIPO IRIS CON IRIS DATASET
"""

# OBTENIENDO LOS DATOS

import numpy as np
import pandas as pd

# Importamos los datos
iris = pd.read_csv("Iris.csv")

# Visualizamos los primeros 5 datos del dataset
#print(iris.head())

# Eliminamos la columna Id ya que pandas crea automaticamente el indice en los dataframes
iris = iris.drop('Id', axis = 1)
print(iris.head())

# ANALIZANDO LOS DATOS

print('Informacion del dataset:')
print(iris.info())

print('Estadisticas del dataset:')
print(iris.describe())

print('Distribucion de datos por especie:')
print(iris.groupby('Species').size())

# VISUALIZAMOS LOS DATOS
import matplotlib.pyplot as plt

# Grafica del Sepalo - Longitud vs Ancho
fig = iris[iris.Species == 'Iris-setosa'].plot(kind = 'scatter',
            x = 'SepalLengthCm', y = 'SepalWidthCm', color = 'blue', label = 'Setosa')
iris[iris.Species == 'Iris-versicolor'].plot(kind = 'scatter',
      x = 'SepalLengthCm', y = 'SepalWidthCm', color = 'green', label = 'Versicolor', ax = fig)
iris[iris.Species == 'Iris-virginica'].plot(kind = 'scatter',
      x = 'SepalLengthCm', y = 'SepalWidthCm', color = 'red', label = 'Virginica', ax = fig)

fig.set_xlabel('Sepalo - Longitud')
fig.set_ylabel('Sepalo - Ancho')
fig.set_title('Sepalo - Longitud vs Ancho')
plt.show()

# Grafica del Petalo - Longitud vs Ancho
fig = iris[iris.Species == 'Iris-setosa'].plot(kind = 'scatter',
            x = 'PetalLengthCm', y = 'PetalWidthCm', color = 'blue', label = 'Setosa')
iris[iris.Species == 'Iris-versicolor'].plot(kind = 'scatter',
      x = 'PetalLengthCm', y = 'PetalWidthCm', color = 'green', label = 'Versicolor', ax = fig)
iris[iris.Species == 'Iris-virginica'].plot(kind = 'scatter',
      x = 'PetalLengthCm', y = 'PetalWidthCm', color = 'red', label = 'Virginica', ax = fig)

fig.set_xlabel('Petalo - Longitud')
fig.set_ylabel('Petalo - Ancho')
fig.set_title('Petalo - Longitud vs Ancho')
plt.show()

# Como este dataset no posee datos perdidos, no se aplican transformaciones

# APLICACION DE ALGORITMOS DE ML

# Con todos los datos del dataset
# Se pueden desarrollar otros dos modelos:
# 1. Solo con los datos del sepalo
# 2. Solo con los datos del petalo

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Construccion de modelo con todos los datos

# Separo las features y el target
X = np.array(iris.drop(['Species'],1))
y = np.array(iris['Species'])

# Separo los datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
print('Son {} datos para entrenamiento y {} datos para prueba'.format(X_train.shape[0], X_test.shape[0]))

# 1. Modelo de Regresion Logistica
algoritmo = LogisticRegression()
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
print('Precision Regresion Logistica: {}'.format(algoritmo.score(X_train, y_train)))

# 2. Maquinas de Soporte Vectorial SVM
algoritmo = SVC()
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
print('Precision Maquinas de Soporte Vectorial: {}'.format(algoritmo.score(X_train, y_train)))

# 3. K Vecinos mas cercanos KNN
algoritmo = KNeighborsClassifier(n_neighbors = 5)
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
print('Precision de K Vecinos mas cercanos: {}'.format(algoritmo.score(X_train, y_train)))

# 4. Arbol de decision
algoritmo = DecisionTreeClassifier()
algoritmo.fit(X_train, y_train)
Y_pred = algoritmo.predict(X_test)
print('Precision del Arbol de descicion: {}'.format(algoritmo.score(X_train, y_train)))
