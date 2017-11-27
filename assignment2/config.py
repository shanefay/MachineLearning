from estimators import Estimator, EstimatorsWithMetrics
import dataset as ds
from sklearn import linear_model, kernel_ridge, svm, tree
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, explained_variance_score, mean_absolute_error, mean_squared_log_error, median_absolute_error
from utils import compose, partial
from math import sqrt
from statistics import median
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
WHITE_WINE = os.path.join('', 'winequality-white.csv')

datasets = {
    'White Wine': ds.white_wine(DATA_DIRECTORY, WHITE_WINE)
}

# The number of datapoints (for each dataset) to use with each estimator
CHUNK_SIZES = [3000]
SMALL_DATASET_SIZE = 3000
MAX_DATASET_SIZE = CHUNK_SIZES[-1]

# Chunk size tolerance
# e.g. With a tolerance of 0.9 a dataset with 9.x million entries will count as a 10 million chunk size
CHUNK_SIZE_TOLERANCE = 0.9

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
    # In cases where no underestimate exist.
    except:
        return 0

# The chosen regression/classification algorithms with chosen metrics
regression = EstimatorsWithMetrics(
    estimators = [
        Estimator('Linear Regression', linear_model.LinearRegression(), SMALL_DATASET_SIZE),
        Estimator('Kernel Regressor', kernel_ridge.KernelRidge(), SMALL_DATASET_SIZE),
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
