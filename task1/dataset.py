import pandas as pd
from sklearn import preprocessing
import os

class Dataset:
    """This class represents a dataset

    Attributes:
        features (matrix): Matrix of feature data
        _regression_target (array): Array of targets for regression algoithms
        _classification_target (array): Array of targets for classification algoithms
    """
    
    def __init__(self, dataframe, features, regression_target, classification_target, map_columns=None):
        """Create a dataset from the given dataframe

        Args:
            dataframe (pandas.DataFrame): Dataframe containing raw data
            features (list of int): Indexes of columns to use as the features
            regression_target (int): Index of the column to use as the regression target
            classification_target (int OR func): Index of the column to use as the
                classification target, OR function to map regression target to produce
                classification target
            map_columns (dict of int to func, optional): Indexes of columns to functions 
                that transform the column data
        """
        if map_columns:
            for index, func in map_columns.items():
                dataframe.iloc[:,index] = dataframe.iloc[:,index].apply(func)

        self.features = dataframe.iloc[:,features]

        self.regression_target = dataframe.iloc[:,regression_target]

        if callable(classification_target):
            self.classification_target = self.regression_target.apply(classification_target)
        else:
            self.classification_target = dataframe.iloc[:,classification_target]

    def size(self):
        """Get the number of datapoints in the dataset.
        
        Returns:
            The number of datapoints in the dataset.
        """
        return len(self.features)


MAX_NROWS = 10000000

def sum_noisy(data_dir, filename):
    print('Loading:', filename)
    filepath = os.path.join(data_dir, filename)
    df = pd.read_csv(filepath, sep=";", nrows=MAX_NROWS)
    features = range(1, 11)
    regression_target = 11
    classification_target = 12
    return Dataset(df, features, regression_target, classification_target)

def sum_clean(data_dir, filename):
    print('Loading:', filename)
    filepath = os.path.join(data_dir, filename)
    df = pd.read_csv(filepath, sep=";", nrows=MAX_NROWS)
    features = range(1, 11)
    regression_target = 11
    classification_target = 12
    return Dataset(df, features, regression_target, classification_target)

def year_predict(data_dir, filename):
    print('Loading:', filename)    
    filepath = os.path.join(data_dir, filename)
    df = pd.read_csv(filepath, nrows=MAX_NROWS)
    features = range(1, 91)
    regression_target = 0
    classification_target_gen = lambda x: str(x)[0:3] + '0s' # convert year to decade
    return Dataset(df, features, regression_target, classification_target_gen)

def new_york_taxi(data_dir, filename):
    print('Loading:', filename)
    filepath = os.path.join(data_dir, filename)
    df = pd.read_csv(filepath, nrows=MAX_NROWS, parse_dates=[2])
    features = [1, 2, 4, 5, 6, 7, 8, 9]
    regression_target = 10
    classification_target_gen = lambda x: x // 60  # convert seconds to minutes
    map_columns = {
        2: (lambda x: x.timestamp()),        # convert datetime to timestamp
        9: (lambda x: 1 if x == 'Y' else 0)  # make store_and_fwd_flag N = 0, Y = 1
    }
    return Dataset(df, features, regression_target, classification_target_gen, map_columns)
