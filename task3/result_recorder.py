import pandas as pd
import numpy as np
import random
from functools import reduce

class ResultRecorder:
    """This class represents the results table.

    This is as specified by the example in the instructions for task 3.
    The `to_csv` method produces a CSV file in the specified format.

    """

    def __init__(self, datasets, estimators, metrics):
        self.tables = {dataset: pd.DataFrame(columns=estimators, index=metrics) for dataset in datasets}

    def add_result(self, dataset, estimator, metric, result):
        self.tables[dataset][estimator][metric] = result

    def to_csv(self, filename):
        # get the ranks (1 highest) of the results in each row
        rank_tables = {dataset: table.rank(axis=1, method='min', ascending=False).astype(np.int64) for dataset, table in self.tables.items()}
        
        combined_rank_table = reduce(lambda x,y: x.append(y), rank_tables.values()) # append tables to count ranks over all datasets
        frequency_table = combined_rank_table.apply(pd.value_counts).fillna(0).astype(np.int64) # count the number of times each rank occurs per column
        frequency_table = frequency_table.transpose().add_prefix('Rank ').transpose().sort_index() # prefix the indexes with 'Rank '

        output = '\n'.join([
            self._tables_to_csv_string(self.tables),
            self._tables_to_csv_string(rank_tables),
            self._dataframe_to_csv_string(frequency_table)
        ])
        with open(filename, 'w') as outfile: outfile.write(output)

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