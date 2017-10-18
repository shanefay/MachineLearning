from config import regression, classification, datasets, CHUNK_SIZES, CHUNK_SIZE_TOLERANCE, OUTPUT_FILE
from result_recorder import ResultRecorder
from estimate_scorer import cross_val_estimate, split_estimate

def score_on_dataset(dataset_name, features, target, estimators, metrics, result_recorder):
    print('Dataset:', dataset_name)
    for estimator in estimators:
        print('\tEstimator:', estimator.name)
        chunks = (chunk for chunk in CHUNK_SIZES 
            if chunk <= len(features) / CHUNK_SIZE_TOLERANCE
            if chunk <= estimator.max_dataset_size)
        for chunk in chunks:
            print('\t\tChunk:', chunk)
            scores = split_estimate(estimator.algorithm, features[:chunk], target[:chunk], metrics)
            for score_name, score in scores.items():
                result_recorder.add_result(estimator.name, dataset_name, score_name, chunk, score)


def main():
    # setup result table column names and indexes
    result_recorder = ResultRecorder(
        CHUNK_SIZES, datasets.keys(), 
        regression.get_estimator_names(), regression.get_metric_names(),
        classification.get_estimator_names(), classification.get_metric_names()
    )

    # run estimate scorer and fill result table
    for dataset_name, dataset in datasets.items():
        score_on_dataset(dataset_name, dataset.features, dataset.regression_target, 
            regression.estimators, regression.metrics, result_recorder)
        score_on_dataset(dataset_name, dataset.features, dataset.classification_target, 
            classification.estimators, classification.metrics, result_recorder)

    # write out results table
    result_recorder.to_csv(OUTPUT_FILE)

if __name__ == '__main__':
    main()