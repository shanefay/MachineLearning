import numpy as np
from sklearn import linear_model, svm, tree
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, precision_score, confusion_matrix
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
    # estimator = svm.LinearSVC()
    metrics = {
        'RMSE': compose(math.sqrt, mean_squared_error),
        'R^2': r2_score
    }

    scores = cross_val_estimate(estimator, sum_noise_data, sum_noise_targets, metrics)
    # scores = split_estimate(estimator, sum_noise_data, sum_noise_targets, metrics);
    print(type(estimator).__name__ + ':')
    print_scores(scores)

def classification(data):
    sum_noise_data = data.iloc[:,1:10]
    sum_noise_targets = data.iloc[:,12]

    # estimator = linear_model.LogisticRegression()
    estimator = tree.DecisionTreeClassifier()
    metrics = {
        'Accuracy': accuracy_score,
        'Precision': partial(precision_score, average='micro'),
    }

    scores = cross_val_estimate(estimator, sum_noise_data, sum_noise_targets, metrics);
    # scores = split_estimate(estimator, sum_noise_data, sum_noise_targets, metrics);
    print(type(estimator).__name__ + ':')
    print_scores(scores)

def print_scores(scores):
    for name, score in scores.items():
        print('{}: {:g}'.format(name, score))
    print()

def main():
    print('Loading data...\n')
    data = pd.read_csv(path, sep=";", nrows=10000)
    print('Loaded', len(data), 'datapoints:')
    print(data.iloc[:,12].value_counts(), '\n')
    regression(data)
    classification(data)


if __name__ == '__main__':
    main()