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

def example():
    basepath = os.path.abspath(os.path.dirname(__file__))
    path = os.path.abspath(os.path.join(basepath, "..", "ML_data/SUM_noise.csv"))
    
    df = pd.read_csv(path, sep=";", nrows=10000)
    features = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5 (meaningless)', 'Feature 6', 'Feature 7', 'Feature 8', 'Feature 9', 'Feature 10']
    regression_target = "Noisy Target"
    classification_target = "Noisy Target Class"
    Dataset(df, features, regression_target, classification_target)

if __name__ == "__main__":
    example()