import numpy as np
from sklearn import datasets, linear_model, svm, tree
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, precision_score
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
import os
import pandas as pd
import math

from utils import compose, partial
from estimate_scorer import *

basepath = os.path.abspath(os.path.dirname(__file__))
path = os.path.abspath(os.path.join(basepath, "..", "ML_data/SUM_noise.csv"))

chunk_size = [100, 500, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000]

def regression(data):
    sum_noise_data = data.iloc[:,1:10]
    sum_noise_targets = data.iloc[:,11]

    estimator = linear_model.LinearRegression()
    metrics = {
        'RMSE': compose(math.sqrt, mean_squared_error),
        'Var': r2_score
    }

    scores = cross_val_estimate(estimator, sum_noise_data, sum_noise_targets, metrics)
    print('\n' + type(estimator).__name__ + ':')
    print(scores)

def classification(data):
    sum_noise_data = data.iloc[:,1:10]
    sum_noise_targets = data.iloc[:,12]

    estimator = linear_model.LogisticRegression()
    metrics = {
        'Accuracy': accuracy_score,
        'Precision': partial(precision_score, average='micro')
    }

    scores = cross_val_estimate(estimator, sum_noise_data, sum_noise_targets, metrics);
    print('\n' + type(estimator).__name__ + ':')
    print(scores)


def main():
    data = pd.read_csv(path, sep=";", nrows=10000)
    print('Running demo...')
    regression(data)
    classification(data)


if __name__ == '__main__':
    main()