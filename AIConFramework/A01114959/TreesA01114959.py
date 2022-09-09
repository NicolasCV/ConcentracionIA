#Nicolas Cardenas Valdez A01114959
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import tree
from matplotlib import pyplot as plt


# # LIMPIEZA

data=pd.read_csv("AIConFramework/A01114959/iris.csv", index_col="Id")
data.head()


#El CSV no contiene NA pero sigue siendo bueno filtrarlos
data=data.dropna().copy()
data.info()

data.duplicated().sum() #Vemos que hay tres duplicados

data.columns

#Podemos ver que hay samples_sizes iguales para cada tipo de especie lo cual es bueno porque mejora nuestra calidad de la 
#informacion
print(data.Species.value_counts())

sns.heatmap(data = data.corr(), annot=True)
plt.show()
#Como podemos ver en el mapa de correlaciones, tenemos unas muy fuertes:
#entre petalwidthcm y sepallengthcm, y sepallengthcm con petallength cm asi que analizaremos estas relaciones

sns.scatterplot(x = 'SepalLengthCm', y = 'SepalWidthCm', data = data, hue = 'Species')
plt.show()

sns.scatterplot(x = 'PetalLengthCm', y = 'PetalWidthCm', data = data ,hue ='Species')
plt.show()
#Como podemos ver en cuanto al sepal tenemos mas variabilidad y hay muchos datos revueltos
#En cuanto a petal vemos que hay una clara uniformidad y un comportamiento que en cuanto supen las medidas del petal 
#en ambas direcciones, vemos que tambien cambia el tipo de especie.
#Tiene muy buena agrupacion lo cual indica que nuestro modelo puede llegar a ser muy preciso


# # MODELO

#Normalmente hariamos lo siguiente:
#data.join(pd.get_dummies(data.Species,prefix="species-"),how="inner").copy()
#data=data.drop(["Id","Species"],axis=1).copy()

#(Esto nos serviria para la regresion lineal)

#Sin embargo, lo que queremos decifrar es el tipo de especie, por lo tanto lo tomamos como la variable dependiente
X = data.drop(['Species'], axis = 1)
X.head()
Y = data['Species']

#Agregamos random state para ser consistentes en la comparacion
x_train,x_test,y_train,y_test=train_test_split(X,Y, test_size=0.25, random_state=0)
x_train.shape,x_test.shape,y_train.shape,y_test.shape

#Usaremos DecisionTreeClassifier debido a que tenemos variables categoricas, queremos decidir una variable categorica a partir de multiples factores numericos, lo cual es perfecto para un DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

#Primero probaremos el modelo sin cambiar ningun parametro
modelo = DecisionTreeClassifier(random_state=0)

modelo.fit(x_train,y_train)

tree.plot_tree(modelo, filled = True)
plt.show()
#Vemos que el arbol muestra muchas decisiones 


## METRICA

#Utilizaremos como metricas el accuracy_score y confusion_matrix de parte de la libreria de sklearn 
#ya que estas nos van a dar una idea de que tan preciso es el modelo (a partir de varias otras metricas) y la matriz
#nos ayuda a determinar como se esta equivocando el modelo y cuanto. Podriamos usar el MSE y otras metricas de errores
#pero el accuracy_score toma en consideracion esto de manera indirecta.
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

#Primero hay que ver cuales features son las mas importantes para el modelo del modelo pasado:
print("Peso Columnas en modelo NO refinado", modelo.feature_importances_)
#Vemos que el factor mas importante es la ultima columna que es "petal width" y no toma en consideracion la primera columna

predicciones=modelo.predict(x_test)

ac = accuracy_score(y_test,predicciones)
print("Accuracy Modelo NO Refinado en test:",ac)

cm = confusion_matrix(y_test, predicciones) 
print(cm)

#Tenemos un excelente accuracy_score lo cual nos indica que nuestro modelo es bueno, la confusion matrix tambien muestra paraledidad, es decir, se esta equivocando parejamente


#Vemos que tenemos es un muy buen modelo pero hay que probar si cambiando los hiperparametros podemos mejorar el accuracy_score

#Primero que nada habra que analizar el arbol del modelo pasado lo cual como podemos ver 4 steps son demasiados
#el ultimo paso consiste de muy poco samples por lo mas seguro es que solo nos este quitando accuracy asi que lo limitaremos a 3 
#para generar mejores decisiones, esto es un ejemplo de sobreajuste. Tambien cambiaremos el min_samples_leaf
#para igualmente evitar que no haya decisiones con samples pequenos que creo que nos estan alterando las decisiones
#Agregamos random state para ser consistentes en la comparacion
modelo2= DecisionTreeClassifier(max_depth = 3, min_samples_leaf=5, random_state=0)

#Tambien dividiremos de nuevo los datos a un split de 80-20 para tener un sample size un poco mas grande para
#ver el nuevo arbol y esperar mejores sample sizes
x_train2,x_test2,y_train2,y_test2=train_test_split(X,Y, test_size=0.2, random_state=0)
x_train.shape,x_test.shape,y_train.shape,y_test.shape

#Fiteamos
modelo2.fit(x_train2,y_train2)

#Checamos de nuevo las importancias
print("Peso Columnas en modelo refinado", modelo2.feature_importances_)
#Vemos que practicamente esta solo considerando la ultima columna lo cual nos indica que podriamos generar
#un muy buen modelo a partir de solo petal-width

#Vemos de nuevo el arbol
tree.plot_tree(modelo2, filled = True)
plt.show()

#Checamos nuestra nueva accuracy con el set de pruebas
predicciones2=modelo2.predict(x_test2)

ac = accuracy_score(y_test2,predicciones2)
print("Accuracy Modelo Refinado en test:",ac)

cm = confusion_matrix(y_test2, predicciones2) 
print(cm)


#Tenemos un accuracy_score de 100%, lo cual un buen numero considerando que no contamos con miles de datos para el entrenamiento.
#Sin embargo esto es para solo el test size
#Nuestra matriz de confusion se reduce a 0 porque no tenemos errores pero tambien en parte porque reducimos el sample size

#Mediremos ahora pero con el set de training solo para tenerlo igualmente de referencia
predicciones2=modelo2.predict(x_train2)

ac = accuracy_score(y_train2,predicciones2)
print("Accuracy Modelo Refinado en train:",ac)

cm = confusion_matrix(y_train2, predicciones2) 
print(cm)
#Vemos que ahora no es 100, significa que nuestro modelo no es perfecto si no que solo es perfecto para los datos test en esta ocasion
#Sin embargo, tenemos todavia un porcentaje de accuracy de casi 97% lo cual significa que no tenemos sobreajuste para solo los datos test

#En conclusion pudimos determinar que modelo de machine learning utilizar (DecisionTreeClasssifier para variables categoricas)
# y armar un modelo base con todos los hiperparametros en default y apartir de lo que pudimos observar del arbol de
#decisiones, poder regularizar esos hiperparametros de entrada para obtener un mejor modelo que no sufre de sobre-ajuste.