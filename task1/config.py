from estimators import Estimator, EstimatorsWithMetrics
import dataset as ds
from sklearn import linear_model, svm, tree
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score
from utils import compose, partial
from math import sqrt
import os

# Output filename
OUTPUT_FILE = 'results.csv'

# Root directory of this project
PROJECT_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')

# Set this as the path to your data directory relative to the project root directory:
# This data directory is assumed to have the same structure as the dropbox dataset folder:
# https://www.dropbox.com/sh/euppz607r6gsen2/AACcVFIxekZXYTEM5ZsMSczEa?dl=0
DATA_DIRECTORY = os.path.join(PROJECT_ROOT, 'ML_data')

# Relative path from data directory to the files containing the datasets.
# The defaults given here match the structure of the dropbox folder:
# https://www.dropbox.com/sh/euppz607r6gsen2/AACcVFIxekZXYTEM5ZsMSczEa?dl=0
SUM_CLEAN = os.path.join('The SUM dataset', 'without noise', 'The SUM dataset, without noise.csv')
SUM_NOISY = os.path.join('The SUM dataset', 'with noise', 'The SUM dataset, with noise.csv')
MILLION_SONG = os.path.join('MillionSong Year-Prediction Dataset (Excerpt)', 'YearPredictionMSD.txt.zip')
NEW_YORK_TAXI = os.path.join('New York City Taxi Trip Duration', 'New York City Taxi Trip Duration.zip')

datasets = {
    'The SUM dataset, without noise': ds.sum_clean(DATA_DIRECTORY, SUM_CLEAN),
    'The SUM dataset, with noise': ds.sum_noisy(DATA_DIRECTORY, SUM_NOISY),
    'YearPredictionMSD': ds.year_predict(DATA_DIRECTORY, MILLION_SONG),
    'New York City Taxi Trip Duration': ds.new_york_taxi(DATA_DIRECTORY, NEW_YORK_TAXI)
}

# The number of datapoints (for each dataset) to use with each estimator
CHUNK_SIZES = [100, 500, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000]
SMALL_DATASET_SIZE = CHUNK_SIZES[-6]
MEDIUM_DATASET_SIZE = CHUNK_SIZES[-3]
MAX_DATASET_SIZE = CHUNK_SIZES[-1]

# Chunk size tolerance
# e.g. With a tolerance of 0.9 a dataset with 9.x million entries will count as a 10 million chunk size
CHUNK_SIZE_TOLERANCE = 0.9


# The chosen regression/classification algorithms with chosen metrics
regression = EstimatorsWithMetrics(
    estimators = [
        Estimator('Linear Regression', linear_model.LinearRegression(), MAX_DATASET_SIZE),
        Estimator('SVM-RBF', svm.SVR(kernel='rbf'), SMALL_DATASET_SIZE)
    ],
    metrics = {
        'RMSE': lambda test, pred: sqrt(mean_squared_error(test, pred)) / pred.mean(),
        'R^2': r2_score
    }
)
classification = EstimatorsWithMetrics(
    estimators = [
        Estimator('Logistic Regression', linear_model.LogisticRegression(), SMALL_DATASET_SIZE),
        Estimator('Descision Tree Classifier', tree.DecisionTreeClassifier(), MEDIUM_DATASET_SIZE)
    ],
    metrics = {
        'Accuracy': accuracy_score,
        'Precision': partial(precision_score, average='macro')
    }
)