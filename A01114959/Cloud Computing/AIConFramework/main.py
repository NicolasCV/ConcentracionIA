from joblib import load
import numpy as np

#Sacamos modelo del analisis y optimizacion que hicimos en el notebook
model = load(R'AIConFramework\A01114959\final.joblib')

inputs = []
for i in range(4):
    input_ = float(input())
    inputs.append(input_)


inputs = np.array(inputs).reshape(-1, 4)

print(model.predict(inputs)[0])