import pandas as pd

class ResultRecorder:
    """This class represents the results table.

    This is as specified by the example in the instructions for task 1.
    The `to_csv` method produces a CSV file in the specified format.

    """
    
    def __init__(self, chunk_sizes, datasets, regression, classfication):
        """Create a new ResultRecorder.
        
        Args:
            chunk_sizes (list): The data chunk sizes.
            datasets (list): The names of the datasets.
            regression (EstimatorClass): The names of the regression algorithms and metrics.
            classfication (EstimatorClass): The names of the classification algorithms and metrics
        """
        indexes = self._get_indexes(datasets, regression) + self._get_indexes(datasets, classfication)
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

    def _get_indexes(self, datasets, estimator_class):
        """ Get all the indexes for regression or classification.
        """
        return [self._get_index(estimator, dataset, metric)
            for estimator in estimator_class.estimators
            for dataset in datasets
            for metric in estimator_class.metrics
        ]
    
    def _get_index(self, estimator, dataset, metric):
        """Get the index in the format as specified in the example.
           format: "$estimator; $dataset; $metric"
           e.g. "Linear_Regression; The SUM Dataset (without noise); RMSE"
        """
        return '; '.join([estimator, dataset, metric])

class EstimatorClass:
    """This class represents a class of estimators with associated metrics.

    e.g. Regression estimators with RMSE and Variance.
    
    Attributes:
        estimators (list): The names of estimator algorithms in this class.
        metrics (list): The names of metrics in this class.
    """
    
    def __init__(self, estimators, metrics):
        self.estimators = estimators
        self.metrics = metrics

def test():
    """A quick (un-automated) test.

    Creates a ResultRecorder, adds a result, and writes to a CSV file.
    """
    chunk_sizes = [100, 500, 1000, 5000]
    datasets = ['Dataset A', 'Dataset B']
    regression = EstimatorClass(['Reg 1', 'Reg 2'], ['Metric X', 'Metric Z'])
    classfication = EstimatorClass(['Class 1', 'Class 2'], ['Metric 1', 'Metric 2'])
    results = ResultRecorder(chunk_sizes, datasets, regression, classfication)
    
    results.add_result(regression.estimators[0], datasets[0], regression.metrics[1], chunk_sizes[2], 12345)
    results.to_csv('test.csv')

if __name__ == '__main__':
    test()