import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

class bruteForceLinearRegression:
    def _init_(self, dataframe=None):
        self.data = dataframe
        self.m = None
        self.b = None
        self.lowestError = None

    def evaluateFunction(self, m, b, x):
        return m * x + b

    def makePrediction(self, x):
        return self.m * x + self.b

    def overview(self):
        funcString = 'The final function is: y = {slope}x + {intercept}'.format(slope=self.m, intercept=self.b)
        print(funcString)
        return (self.m,self.b)

    def showData(self):
        plt.scatter(self.data[['x']], self.data[['y']])
        
        if (self.m and self.b):
            predictions = []
            for x in self.data[['x']].to_numpy():
                predictions.append(self.makePrediction(x[0]))
            plt.plot(self.data[['x']],predictions, '--')
        
        plt.show()

    #NUMERIC LINEAR REGRESSION
    def mean(self, values): 
        return sum(values) / float(len(values)) 

    def variance(self, values, mean): 
        return sum([(x-mean)**2 for x in values]) 
    
    def covariance(self, x, y): 
        mediaX = self.mean(x)
        mediaY = self.mean(y)
        covar = 0.0 
        for i in range(len(x)): 
            covar += (x[i] - mediaX) * (y[i] - mediaY) 
        return covar 

    def coefficients(self): 
        x = self.data[['x']].to_numpy()
        y = self.data[['y']].to_numpy()
        b1 = self.covariance(x,y) / self.variance(x, self.mean(x)) 
        b0 = self.mean(y) - (b1 * self.mean(x))
        return [b0[0], b1[0]] 

    def numericLinearRegression(self):
        self.b, self.m = self.coefficients()
        self.overview()

    ########################################################
    
    #BRUTE FORCE LINEAR REGRESSION (NOT ACCURATE)
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

    def manualRegression(self, step):
        
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
        
                
