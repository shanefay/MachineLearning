import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import csv
import os.path

basepath = os.path.abspath(os.path.dirname(__file__))
path = os.path.abspath(os.path.join(basepath, "..", "ML_data/SUM_noise.csv"))
#with open(path) as f:
#    test = [line for line in csv.reader(f)]

sum_noise = open(path, 'r')
sum_noise_data = np.loadtxt(sum_noise, dtype='int', delimiter=';', usecols=(1,2,3,4,6,7,8,9,10), skiprows=1)
sum_noise_targets = np.loadtxt(sum_noise, dtype='int', delimiter=';', usecols=(11), skiprows=1)
chunk_size = [100, 500, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000]

sum_X = sum_noise_data

print(sum_noise_targets[0])























	