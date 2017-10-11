from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, cross_validate

def split_estimate(estimator, X, y, metrics, test_size=0.3):
    """Score an estimated model using a simple data split.

    A 70/30 split of training to testing data is used by default.

    Args:
        estimator: Scikit-learn model estimator.
        X: Feature data matrix.
        y: Target data array.
        metrics: Metrics to be used for scoring the estimated model.
            A Dictionary of metric names to metric evaluation functions.
            e.g. {'accuracy': scklearn.metrics.accuracy_score, etc.}
        test_size (optional): The proportion of the data to be used for testing.
    
    Returns:
        A dictionary of metric names to scores.
        The scores are the metrics on the test targets verses predicted targets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    return {name: metric(y_test, y_pred) for name, metric in metrics.items()}

def cross_val_estimate(estimator, X, y, metrics, k_fold=10):
    """Score an estimated model using k-fold cross validation.

    10-fold cross validation is used by default.
    
    Args:
        estimator: Scikit-learn model estimator.
        X: Feature data matrix.
        y: Target data array.
        metrics: Metrics to be used for scoring the estimated model.
            A Dictionary of metric names to metric evaluation functions.
            e.g. {'accuracy': scklearn.metrics.accuracy_score, etc.}
        test_size (optional): The proportion of the data to be used for testing.
    
    Returns: 
        A dictionary of metric names to scores.
        The scores are the metrics on the k-fold test targets verses predicted 
        targets. As there are k-fold test sets, the mean of each metric is 
        taken as the scores.
    """
    scoring = {name: make_scorer(metric) for name, metric in metrics.items()}
    scores = cross_validate(estimator, X, y, cv=k_fold, scoring=scoring)
    return {name: scores['test_' + name].mean() for name in scoring}
