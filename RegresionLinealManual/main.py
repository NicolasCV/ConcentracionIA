from bruteForceLinearRegression import bruteForceLinearRegression as bflr

model = bflr()
model.manualRegression('data.csv', 0.4)
model.overview()
res = model.evaluateModel(123)