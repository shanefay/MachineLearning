
class Estimator:
    """A machine learning model estimator
    
    Attributes:
        name: Name of the estimator
        algorithm: Estimator algorithm
    """

    def __init__(self, name, algorithm):
        self.name = name
        self.algorithm = algorithm

class EstimatorsWithMetrics:
        
    def __init__(self, estimators, metrics):
        self.estimators = estimators
        self.metrics = metrics

    def get_estimator_names(self):
        return [estimator.name for estimator in self.estimators] # + ["time"]

    def get_metric_names(self):
        return self.metrics.keys()