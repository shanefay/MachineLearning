import pandas as pd
import os

class Dataset:
    def __init__(self, dataframe, features, regression_target, classification_target):
        self.features = features
        self.targets = Targets(regression_target, classification_target)

class Targets:
    def __init__(self, regression, classifictation):
        self.regression = regression
        self.classifictation = classifictation

def Sum_Noise():
    basepath = os.path.abspath(os.path.dirname(__file__))
    path = os.path.abspath(os.path.join(basepath, "..", "ML_data/SUM_noise.csv"))
    
    df = pd.read_csv(path, sep=";", nrows=10000)
    features = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5 (meaningless)', 'Feature 6', 'Feature 7', 'Feature 8', 'Feature 9', 'Feature 10']
    regression_target = df.iloc[:12]
    classification_target = df.iloc[:13]
    Dataset(df, features, regression_target, classification_target)

def Sum_clean(): 
    basepath = os.path.abspath(os.path.dirname(__file__))
    path = os.path.abspath(os.path.join(basepath, "..", "ML_data/SUM_clean.csv"))
    
    df = pd.read_csv(path, sep=";", nrows=10000)
    features = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5 (meaningless)', 'Feature 6', 'Feature 7', 'Feature 8', 'Feature 9', 'Feature 10']
    regression_target = df.iloc[:12]
    classification_target = df.iloc[:13]
    Dataset(df, features, regression_target, classification_target)     

def Year_Predict(): 
    basepath = os.path.abspath(os.path.dirname(__file__))
    path = os.path.abspath(os.path.join(basepath, "..", "ML_data/YearPredictionMSD.txt"))
    
    df = pd.read_csv(path, sep=",", nrows=10000, header=None)
    features = df.iloc[:,1:90]
    regression_target = df.iloc[:,0]
    classification_target = df.iloc[:,0].apply(lambda x: str(x)[0:3] + '0s')
    print(classification_target[0:50])
    Dataset(df, features, regression_target, classification_target)    

def Taxi(): 
    basepath = os.path.abspath(os.path.dirname(__file__))
    path = os.path.abspath(os.path.join(basepath, "..", "ML_data/train.csv"))
    
    df = pd.read_csv(path, sep=",", nrows=10000)
    features = pd.merge(df.iloc[0:2], df.iloc[4:9])
    regression_target = df.iloc[:10]
    classification_target_seconds = df.iloc[:10]
    """ // forces interger division - only whole minutes"""
    classification_target = int(classification_target_seconds) // 60
    Dataset(df, features, regression_target, classification_target)       

if __name__ == "__main__":
    Taxi()