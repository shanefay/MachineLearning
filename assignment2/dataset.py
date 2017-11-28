import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

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
    df = pd.read_csv(filepath, nrows=MAX_NROWS,delimiter=";",header=0)
    # Assuming same lines from your example
    df.columns = ["fixed acidity","volatile acidity","citric acid",
                "residual sugar","chlorides","free sulfur dioxide",
                "total sulfur dioxide","density","pH","sulphates",
                "alcohol","quality"]
    output_text_data(df, 'original_data')
    do_plots(df, 'original_data')
    cols_to_norm = ["fixed acidity","volatile acidity","citric acid",
                "residual sugar","chlorides","free sulfur dioxide",
                "total sulfur dioxide","density","pH","sulphates",
                "alcohol"]
    df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    #do_plots(df, 'normalized')
    print('Feature plots saved')
    features = range(0,10)
    regression_target = 11
    classification_target = 0
    return Dataset(df, features, regression_target, classification_target)

def do_plots(dataframe, name):
    do_histogram(dataframe, name)
    do_scatterplots(dataframe)
    do_correlations(dataframe, name)

def output_text_data(dataframe, name):
    text_file = open(name + "_details.txt", "w")
    text_file.write(str(dataframe.describe()))
    text_file.close()

def do_scatterplots(df):
    for col in df.columns:
        df.plot(kind='scatter', x=col, y='quality').get_figure().savefig(col + '_vs_quality_scatterplot')

def do_histogram(df, name):
    return None
    # for col in df.columns:
    #     df.hist(column=col, bins=10).plot().get_figure().savefig(col + '_histogram')
    
def do_correlations(df, name):
    correlations = df.corr()
    # plot correlation matrix
    fig2 = plt.figure(figsize=(14, 14))
    ax2 = fig2.add_subplot(111)
    cax = ax2.matshow(correlations, vmin=-1, vmax=1)
    fig2.colorbar(cax)
    ticks = np.arange(0,12,1)
    ax2.set_xticks(ticks)
    ax2.set_yticks(ticks)
    ax2.set_xticklabels(df.columns, rotation=90)
    ax2.set_yticklabels(df.columns)
    plt.savefig(name + "_feature_correlations.png")
    plt.close()