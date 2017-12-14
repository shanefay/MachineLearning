import pandas as pd
import numpy as np
import random
from functools import reduce

class ResultRecorder:
    """This class represents the results table.

    The `to_csv` method produces a CSV file containing the results.

    """

    def __init__(self, datasets, estimators, metrics):
        self.tables = {dataset: pd.DataFrame(columns=estimators, index=metrics) for dataset in datasets}

    def add_result(self, dataset, estimator, metric, result):
        self.tables[dataset][estimator][metric] = result

    def to_csv(self, filename):
        with open(filename, 'w') as outfile: 
            outfile.write(self._tables_to_csv_string(self.tables))

    def _tables_to_csv_string(self, tables):
        suffixed_tables = [table.add_suffix(' ({})'.format(dataset)) for dataset, table in tables.items()] # add name of dataset to column labels
        joined_tables = pd.concat(suffixed_tables, axis=1) # join tables side by side
        return self._dataframe_to_csv_string(joined_tables)

    def _dataframe_to_csv_string(self, dataframe):
        return dataframe.to_csv(sep=';', na_rep='#N/A', encoding='utf-8')

    
def test():
    dss = ['ds1', 'ds2']
    algs = ['lin', 'tree']
    mets = ['average', 'accuracy']

    rr = ResultRecorder(dss, algs, mets)

    for ds in dss: 
        for alg in algs: 
            for met in mets:
                rr.add_result(ds, alg, met, random.randint(1, 100))

    rr.to_csv('test.csv')

if __name__ == '__main__':
    test()