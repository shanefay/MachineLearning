from config import regression, datasets, CHUNK_SIZE, OUTPUT_FILE
from result_recorder import ResultRecorder
from estimate_scorer import cross_val_estimate, split_estimate
from timeit import default_timer as timer

def score_on_dataset(dataset_name, features, target, estimators, metrics, result_recorder):
    print('Dataset:', dataset_name)
    for estimator in estimators:
        print('\tEstimator:', estimator.name)
        start_time = timer()
        # split estimate necessary for our machines to run this in time, feel free to change to cross_val_estimate
        scores = cross_val_estimate(estimator.algorithm, features[:CHUNK_SIZE], target[:CHUNK_SIZE], metrics)
        for score_name, score in scores.items():
           result_recorder.add_result(dataset_name, estimator.name, score_name, score)
           print (score_name, score)

        result_recorder.add_result(dataset_name, estimator.name, "time", timer() - start_time)
        print("TIME TAKEN = " + str(timer() - start_time))

def main():
    # setup result table column names and indexes
    result_recorder = ResultRecorder(
        datasets.keys(), 
        regression.get_estimator_names(),
        regression.get_metric_names()
    )

    # run estimate scorer and fill result table
    for dataset_name, dataset in datasets.items():
        score_on_dataset(dataset_name, dataset.features, dataset.regression_target, 
            regression.estimators, regression.metrics, result_recorder)

    # write out results table
    result_recorder.to_csv(OUTPUT_FILE)

if __name__ == '__main__':
    main()