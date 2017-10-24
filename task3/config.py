from estimators import Estimator, EstimatorsWithMetrics
import dataset as ds
from sklearn import linear_model, svm, tree
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, explained_variance_score, mean_absolute_error, mean_squared_log_error, median_absolute_error
from utils import compose, partial
from math import sqrt
from statistics import median
import timeit
import os


def median_overestimate(predY, trueY):
    diffs = predY - trueY
    try:
        return median(filter(lambda x: x >= 0, diffs))
    # In cases where no overestimates exist.
    except:
        return 0
def median_underestimate(predY, trueY):
    diffs = predY - trueY
    try:
        return median(filter(lambda x: x <= 0, diffs))
    # In cases where no overestimates exist.
    except:
        return 0

# Output filename
OUTPUT_FILE = 'results.csv'

# Set this as the path to your data directory
DATA_DIRECTORY = os.path.join(os.path.abspath(os.path.dirname(__file__)),'..','ML_data')

# Relative path from data directory to the files containing the datasets.
# The defaults given here match the structure of the dropbox folder:
# https://www.dropbox.com/sh/euppz607r6gsen2/AACcVFIxekZXYTEM5ZsMSczEa?dl=0

MILLION_SONG = os.path.join('YearPredictionMSD.txt.zip')
NEW_YORK_TAXI = os.path.join('New York City Taxi Trip Duration.zip')

datasets = {
    'YearPredictionMSD': ds.year_predict(DATA_DIRECTORY, MILLION_SONG),
    'New York City Taxi Trip Duration': ds.new_york_taxi(DATA_DIRECTORY, NEW_YORK_TAXI)
}

# The number of datapoints (for each dataset) to use with each estimator
CHUNK_SIZES = [1000]
SMALL_DATASET_SIZE = 1000
MAX_DATASET_SIZE = CHUNK_SIZES[-1]

# Chunk size tolerance
# e.g. With a tolerance of 0.9 a dataset with 9.x million entries will count as a 10 million chunk size
CHUNK_SIZE_TOLERANCE = 0.9

wrapper1 = timeit.timeit(linear_model.LinearRegression())

# The chosen regression/classification algorithms with chosen metrics
regression = EstimatorsWithMetrics(
    estimators = [
        Estimator('Linear Regression', wrapper1, MAX_DATASET_SIZE),
        Estimator('Hubor Regressor', linear_model.HuberRegressor(), SMALL_DATASET_SIZE),
        Estimator('Perceptron', linear_model.Perceptron(), SMALL_DATASET_SIZE),
        Estimator('Linear Support Vector Machine', svm.LinearSVR(), SMALL_DATASET_SIZE),
        Estimator('Decision Tree', tree.DecisionTreeRegressor(), SMALL_DATASET_SIZE),
    ],
    metrics = {
        'RMSE': compose(sqrt, mean_squared_error),
        'R^2': r2_score,
        'Explained Variance':explained_variance_score,
        'Mean Absolute Error':mean_absolute_error, 
        'Median Absolute Error':median_absolute_error,
        'Median Overestimate':median_overestimate,
        'Median Underestimate':median_underestimate,
        'Max Overestimate': lambda trueY, predY : max(predY - trueY),
        'Min Underestimate': lambda trueY, predY : min(predY - trueY)
    }
)
classification = EstimatorsWithMetrics(
    estimators = [
        Estimator('Logistic Regression', linear_model.LogisticRegression(), SMALL_DATASET_SIZE),
        Estimator('Decision Tree Classifier', tree.DecisionTreeClassifier(), MAX_DATASET_SIZE)
    ],
    metrics = {
        'Accuracy': accuracy_score,
        'Precision': partial(precision_score, average='micro')
    }
)
