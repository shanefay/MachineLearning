from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score
from utils import compose, partial
from math import sqrt

class Metrics:
    regression = {
        'RMSE': compose(sqrt, mean_squared_error),
        'R^2': r2_score
    }

    classification = {
        'Accuracy': accuracy_score,
        'Precision': partial(precision_score, average='micro')
    }