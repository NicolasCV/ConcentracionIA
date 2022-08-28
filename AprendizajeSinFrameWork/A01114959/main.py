from bruteForceLinearRegression import bruteForceLinearRegression as bflr

model = bflr()
model.loadData(R'C:\Users\omega\OneDrive\Documents\GitHub\ConcentracionIA\AprendizajeSinFrameWork\A01114959\data.csv')
model.numericLinearRegression()
model.showData()