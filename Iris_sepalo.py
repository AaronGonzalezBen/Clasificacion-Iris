"""
CLASIFICADOR FLORES TIPO IRIS CON IRIS DATASET A PARTIR DEL SEPALO
"""

# OBTENIENDO LOS DATOS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Importamos los datos
iris = pd.read_csv("Iris.csv")

# Eliminamos la columna Id ya que pandas crea automaticamente el indice en los dataframes
# Adicionalmente eliminamos las columnas de Petalo para predecir los valores solo a partir de los Sepalos
iris = iris.drop(['Id','PetalLengthCm','PetalWidthCm'], axis = 1)
print(iris.head())

# ANALIZANDO LOS DATOS

print('Informacion del dataset:')
print(iris.info())

print('Estadisticas del dataset:')
print(iris.describe())

print('Distribucion de datos por especie:')
print(iris.groupby('Species').size())

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

# APLICACION DE LOS ALGORITMOS DE ML

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
print('Precision de Arbol de Decision: {}'.format(algoritmo.score(X_train, y_train)))