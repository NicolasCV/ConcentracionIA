#Nicolas Cardenas Valdez A01114959
import pandas as pd

data=pd.read_csv(R"C:\Users\omega\Downloads\iris.csv", index_col="Id")
data.head()

#El CSV no contiene NA pero sigue siendo bueno filtrarlos
data=data.dropna().copy()
data.info()

print(data.duplicated().sum()) #Vemos que hay tres duplicados
print(data.columns)
print(data.Species.value_counts())

#Normalmente hariamos lo siguiente:
#data.join(pd.get_dummies(data.Species,prefix="species-"),how="inner").copy()
#data=data.drop(["Id","Species"],axis=1).copy()

#Sin embargo, lo que queremos decifrar es el tipo de especie, por lo tanto lo tomamos como la variable dependiente
X = data.drop(['Species'], axis = 1)
Y = data['Species']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y, test_size=0.25,random_state=50)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

#Usaremos DecisionTreeClassifier debido a que tenemos variables categoricas
from sklearn.tree import DecisionTreeClassifier
modelo = DecisionTreeClassifier(random_state=0)
modelo.fit(x_train.values,y_train.values)

from sklearn import tree
from matplotlib import pyplot as plt
tree.plot_tree(modelo)
plt.show()

from sklearn.metrics import accuracy_score
predicciones=modelo.predict(x_test.values)
print(accuracy_score(y_test.values,predicciones))
#Tenemos un excelente accuracy_score lo cual nos indica que nuestro modelo es bueno

from joblib import dump
dump(modelo, 'modeloIris.joblib') 

