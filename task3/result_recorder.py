import pandas as pd

class ResultRecorder:
    """This class represents the results table.

    This is as specified by the example in the instructions for task 1.
    The `to_csv` method produces a CSV file in the specified format.

    """
    
    def __init__(self, chunk_sizes, datasets, 
                 regression_estimators, regression_metrics, 
                 classfication_estimators, classfication_metrics):
        """Create a new ResultRecorder.
        
        Args:
            chunk_sizes (list): The data chunk sizes.
            datasets (list): The names of the datasets.
            regression_estimators (list): The names of the regression algorithms.
            regression_metrics (list): The names of the regression metrics.
            classfication_estimators (list): The names of the classfication algorithms.
            classfication_metrics (list): The names of the classfication metrics.
        """
        indexes = (self._get_indexes(regression_estimators, datasets, regression_metrics) +
                   self._get_indexes(classfication_estimators, datasets, classfication_metrics))
        self._record = pd.DataFrame(index=indexes, columns=chunk_sizes)

    def add_result(self, estimator, dataset, metric, chunk_size, result):
        """Add a result corresponding to the score of an estimated model.
        
        Args:
            estimator: The name of the estimator algorithm.
            dataset: The name of the dataset used for training and testing
            metric: The name of the metric used for scoring.
            chunk_size: The number of datapoints in the dataset.
            result: The score for the estimated model.
        """
        index = self._get_index(estimator, dataset, metric)
        self._record[chunk_size][index] = result

    def to_csv(self, filename):
        """Write out a CSV with the recorded results.
        
        The output CSV file is in the format specified by the task instructions. 

        Args:
            filename: The name of the file to be written out to.
        """
        self._record.to_csv(filename, na_rep='#N/A', encoding='utf-8')

    def _get_indexes(self, estimators, datasets, metrics):
        """ Get all the indexes for regression or classification.
        """
        return [self._get_index(estimator, dataset, metric)
            for estimator in estimators
            for dataset in datasets
            for metric in metrics
        ]
    
    def _get_index(self, estimator, dataset, metric):
        """Get the index in the format as specified in the example.
           format: "$estimator; $dataset; $metric"
           e.g. "Linear_Regression; The SUM Dataset (without noise); RMSE"
        """
        return '; '.join([estimator, dataset, metric])

def test():
    """A quick (un-automated) test.

    Creates a ResultRecorder, adds a result, and writes to a CSV file.
    """
    chunk_sizes = [100, 500, 1000, 5000]
    datasets = ['Dataset A', 'Dataset B']
    regression_estimators = ['Reg 1', 'Reg 2']
    regression_metrics = ['Metric X', 'Metric Z']
    classfication_estimators = ['Class 1', 'Class 2']
    classfication_metrics = ['Metric 1', 'Metric 2']
    results = ResultRecorder(chunk_sizes, datasets, regression_estimators, regression_metrics,
                             classfication_estimators, classfication_metrics)
    
    results.add_result(regression_estimators[0], datasets[0], regression_metrics[1], chunk_sizes[2], 12345)
    results.to_csv('test.csv')

if __name__ == '__main__':
    test()