import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import accuracy_score
import os
import pandas as pd

basepath = os.path.abspath(os.path.dirname(__file__))
path = os.path.abspath(os.path.join(basepath, "..", "ML_data/SUM_noise.csv"))


data = pd.read_csv(path, sep=";")

#figure out if we need feature 5
sum_noise_data = data.iloc[:,[1,2,3,4,6,7,8,9,10]]
sum_noise_targets = data.iloc[:,12]

chunk_size = [100, 500, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000]

print(sum_noise_targets[0])

sum_X_train = sum_noise_data[:100] #insert chunk here if we loop
sum_X_test = sum_noise_data[-100:]

sum_y_train = sum_noise_targets[:100]
sum_y_test = sum_noise_targets[-100:]

regr = linear_model.LogisticRegression()
# Train the model using the training sets
regr.fit(sum_X_train, sum_y_train)

# Make predictions using the testing set
sum_y_pred = regr.predict(sum_X_test)
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Accurary: %.2f" % accuracy_score(sum_y_test, sum_y_pred))
# Explained variance score: 1 is perfect prediction

