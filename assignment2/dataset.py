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

def white_wine(data_dir, filename):
    print('Loading:', filename)
    filepath = os.path.join(data_dir, filename)
    df = pd.read_csv(filepath, nrows=MAX_NROWS,delimiter=";")
    normalized = preprocessing.normalize(df)
    df = pd.DataFrame(normalized)
    features = range(0,10)
    regression_target = 11
    classification_target = 0
    return Dataset(df, features, regression_target, classification_target)  