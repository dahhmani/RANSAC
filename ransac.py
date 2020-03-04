import numpy as np
import math
import random
import csv
import matplotlib.pyplot as plt
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--fname', required=False, help='Input csv file containing data',
            default='data.csv', type=str)
    ap.add_argument('-N', '--nsamples', required=False, help='Number of samples to select randomly',
            default='10000', type=int)
    ap.add_argument('-t', '--threshold', required=False, help='Distance threshold to identify outliers',
            default='25', type=int)
    ap.add_argument('-d', '--polydegree', required=False, help='Degree of the polynomial model',
            default='2', type=int)
    args = vars(ap.parse_args())
    
    x, y = loadDataset(args['fname'])
    
    # model, yest = polyFit(x, y, args['polydegree'])
    model, yest, outliersPercentage = polyRansacFit(x, y, args['polydegree'], args['nsamples'], args['threshold'])
    print('model parameters =', model)
    print('outliers percentage =', outliersPercentage)

    plt.plot(x, y, 'bo', label='Input Data')
    plt.plot(x, yest, 'r', label='RANSAC Polynomial Fit', linewidth=4)
    plt.axis([0,500,-100,400])
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(args['fname'])
    plt.show()

def polyRansacFit(x_dataset, y_dataset, degree, no_samples, distanceThreshold):
    ''' Robust Estimation (Outlier Rejection Technique) '''

    datasetSize = len(y_dataset)
    min_no_outliers = math.inf
    # visitedSamples = set()
    X_dataset = constructRegressionMatrix(x_dataset, degree) 

    for _ in range(no_samples):
    # 1/ sample minimum number of data points
        sampleIndices = random.sample(range(datasetSize), degree+1)
        # while sampleIndices in visitedSamples:
        #     sampleIndices = tuple(random.sample(range(datasetSize), degree+1))
        # visitedSamples.add(sampleIndices)
        X_sample, y_sample = [], []
        for i in sampleIndices:
            X_sample.append(X_dataset[i,:])
            y_sample.append(y_dataset[i])
    # 2/ fit a linear model (polynomial)
        p = np.linalg.inv(X_sample).dot(y_sample) # model parameters
    # 3/ compute the number of outliers (data loss)
        no_outliers = countOutliers(X_dataset, y_dataset, p, distanceThreshold)
    # 4/ update the model's parameters if necessary
        if no_outliers < min_no_outliers:
            min_no_outliers = no_outliers
            bestModel = p 

    # estimate the model output over all data points
    yest_dataset = X_dataset.dot(bestModel)

    return (bestModel, yest_dataset, min_no_outliers/datasetSize)

def polyFit(x_dataset, y_dataset, degree):
    ''' Ordinary Least Squares Polynomial Regression (X.p = yest ~ y) '''
    
    X_dataset = constructRegressionMatrix(x_dataset, degree) 

    # fit a linear least squares model (polynomial) by solving the normal equations
    p = np.linalg.inv(X_dataset.T.dot(X_dataset)).dot(X_dataset.T).dot(y_dataset) # = pinv(X_dataset).y_dataset
    # p = np.polyfit(x_dataset,y_dataset,degree)

    # estimate the model output over all data points
    yest_dataset = X_dataset.dot(p)

    return (p, yest_dataset)

def constructRegressionMatrix(x_dataset, degree):
    X_dataset = np.ones((len(x_dataset),1))
    for i in range(1,degree+1):
        X_dataset = np.column_stack((x_dataset**i, X_dataset))

    return X_dataset

def countOutliers(X_dataset, y_dataset, p, distanceThreshold):
    exampleLosses = abs(X_dataset.dot(p) - y_dataset) # vertical distance between estimate and ground truth
    outliersCount = 0 
    for loss in exampleLosses:
        if loss > distanceThreshold:
            outliersCount += 1
    
    return outliersCount # number of outliers can be viewed as the data loss

def loadDataset(fileName):
    x, y = [], []
    with open(fileName) as csvfile:
    	data = csv.reader(csvfile, delimiter=',')
    	for row in data:
            x.append(row[0])
            y.append(row[1])
    x = np.array(x[1:], dtype=np.float32)   
    y = np.array(y[1:], dtype=np.float32)   

    return (x, y)

if __name__== '__main__':
    main()