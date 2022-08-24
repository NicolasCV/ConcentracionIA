import pandas as pd
import numpy as np

class bruteForceLinearRegression:
    def _init_(self, dataframe=None):
        self.data = dataframe
        self.m = None
        self.b = None
        self.lowestError = None
        
    def evaluateFunction(self, m, b, x):
        return m * x + b

    def evaluateModel(self, x):
        return self.m * x + self.b

    def getError(self, m, b, point):
        x = point[0]
        y = point[1]
        return abs(y - self.evaluateFunction(m, b, x))

    def getTotalMSE(self, m, b):
        totalSE = 0
        for index, row in self.data.iterrows():
            totalSE += (self.getError(m, b, (row['x'],row['y'])))**2
        
        return totalSE/len(self.data)

    def getSlopeIntervals(self, step):
        maxSlope = self.getBiggestSlope()[0]
        range_ = self.getRange()[0]   
        return list(np.arange(maxSlope - range_, maxSlope + range_, step))

    def getInterceptIntervals(self, step):
        multCoefficient = 10
        lowEndIntercept = -multCoefficient*(max(self.data[['y']].to_numpy())+1)
        highEndIntercept = multCoefficient*(max(self.data[['y']].to_numpy())+1)
        return list(np.arange(lowEndIntercept, highEndIntercept, step))


    def loadData(self, filename):
        columns = ['x','y']
        self.data = pd.read_csv(filename)
        self.data.columns = columns

    def getBiggestSlope(self):
        slope = float("-inf")
        y = self.data[['y']].to_numpy()
        for i in range(1,len(y)):
            slope = max(slope, abs(y[i] - y[i-1]))
        return slope

    def getRange(self):
        y = self.data[['y']].to_numpy()
        return abs(max(y) - min(y))


    def manualRegression(self, filename, step):
        data = self.loadData(filename)
        
        lowestMSE = float("inf")
        optimalSlope = 0
        optimalIntercept = 0

        slopeRange = self.getSlopeIntervals(step)
        interceptRange = self.getInterceptIntervals(step)

        for m in slopeRange:
            for b in interceptRange:
                
                error = self.getTotalMSE(m, b)
                
                if error < lowestMSE:
                    optimalSlope = m
                    optimalIntercept = b
                    lowestMSE = error

        self.m = optimalSlope
        self.b = optimalIntercept
        self.lowestError = lowestMSE
        
    def overview(self):
        funcString = 'The final function is: y = {slope}x + {intercept} with an MSE of: {error}'.format(slope=self.m, intercept=self.b, error = self.lowestError)
        print(funcString)
        return (self.m,self.b)
                
