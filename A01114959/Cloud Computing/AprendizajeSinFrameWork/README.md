El archivo bruteForceLinearRegression contiene una clase para crear una regresion lineal sin ayuda de ningun tipo de framework. Esta disenada para solamente dos variables. X y Y.

Para el ejemplo que se encuentra en main.py usamos un dataset de informacion de un hospital.
El x o la variable independiente es la edad del paciente y la y variable dependiente es cuanto pago en dicho hospital. (En dolares)

El primer paso que se debe hacer es cargar la informacion con loadData() y despues hacemos la regresion lineal para determinar los coeficientes para el intercepto y el coeficiente del efecto de x. Sin embargo, esto no muestra ninguna informacion.

Finalmente, con showData() mostramos la formula en la consola, de como quedaria para poder hacer predicciones. Pero la clase tambien ya cuenta con una funcion para hacer predicciones utilizando el modelo que se armo. Para esto se hizo model.predict(x) y se le mete como parametro x la x que deseamos predecir con nuestro modelo. La grafica que muestra showData() es un scatterplot de toda la info y una linea dasheada mostrando la funcion que sacamos de nuestra regresion lineal.

La regresion lineal se saca de una manera numerica utilizando la varianza y la covarianza. Es un metodo tradicional implementado desde cero con python.