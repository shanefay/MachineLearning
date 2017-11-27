
class Estimator:
    """A machine learning model estimator
    
    Attributes:
        name: Name of the estimator
        algorithm: Estimator algorithm
        max_dataset_size: Maximum number of datapoints that should be 
                          used with this estimator 
    """

    def __init__(self, name, algorithm, max_dataset_size):
        self.name = name
        self.algorithm = algorithm
        self.max_dataset_size = max_dataset_size

class EstimatorsWithMetrics:
        
    def __init__(self, estimators, metrics):
        self.estimators = estimators
        self.metrics = metrics

    def get_estimator_names(self):
        return [estimator.name for estimator in self.estimators] # + ["time"]

    def get_metric_names(self):
        return self.metrics.keys()