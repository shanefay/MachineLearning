import plotlib.pyplot as plt
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
#figure out if we need feature 5
sum_noise_data = np.loadtxt(sum_noise, dtype='int', delimiter=';', usecols=(1,2,3,4,6,7,8,9,10), skiprows=1)
sum_noise_targets = np.loadtxt(sum_noise, dtype='int', delimiter=';', usecols=(11), skiprows=1)
chunk_size = [100, 500, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000]

sum_X = sum_noise_data

print(sum_noise_targets[0])

sum_X_train = sum_noise_data[:100] #insert chunk here if we loop
sum_X_test = sum_noise_data[-100:]

sum_y_train = sum_noise_targets[:100]
sum_y_test = sum_noise_targets[-100:]


regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(sum_X_train, sum_y_train)

# Make predictions using the testing set
sum_y_pred = regr.predict(sum_X_test)	



# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(sum_y_test, sum_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(sum_y_test, sum_y_pred))

# Plot outputs
plt.scatter(sum_X_test, sum_y_test,  color='black')
plt.plot(sum_X_test, sum_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()