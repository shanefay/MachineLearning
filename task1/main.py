import utils
import estimate_scorer as es
import result_recorder as rr
import dataset as ds
import estimators
import metrics
"""import demo"""


chunk_size = [100, 500, 5000]
""", 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000"""
data_sets = {'SUM_clean': ds.sum_clean()}
""", ds.sum_noisy(), ds.year_predict(), ds.new_york_taxi()"""
"""demo.regression(dataset.sum_clean())"""

"""for algorithm
	for dataset
		for metric
			for j in chunk_size.length
				for estimate_scorer
					es.split_estimate(, , , , chunk_size[j])
					es.cross_val_estimate(, , , , chunk_size[j])"""

data = ds.sum_clean()
estimator1 = estimators.Estimators.regression.get('Linear Regression')
metrics1 = metrics.Metrics.regression
thing = es.split_estimate(estimator1, data.features[:500], data.regression_target[:500], metrics1)

results = rr.ResultRecorder(chunk_size, ['SUM_clean'], ['Linear Regression'], metrics1.keys(), [], [])
results.add_result("Linear Regression", 'data', 'RMSE', 500, thing['RMSE'])
results.to_csv('test.csv')

new_results = rr.ResultRecorder(chunk_size,
	['SUM_clean', 'SUM_noisy', 'Million Songs', 'New York Taxis'],
	estimators.Estimators.regression.keys(), 
	metrics.Metrics.regression.keys(),
	metrics.Metrics.classification.keys(),
	estimators.Estimators.classification.keys())

print(estimators.Estimators.regression.keys())
for estimator in estimators.Estimators.regression.keys():
	for data_set in data_sets.keys(): 
		for metric in metrics.Metrics.regression.keys():
			for chunk in chunk_size:
				if chunk < len(data_sets[data_set].features):
					 result = es.split_estimate(estimators.Estimators.regression[estimator], 
					 	data_sets[data_set].features[:chunk], 
					 	data_sets[data_set].regression_target[:chunk], 
					 	metrics.Metrics.regression)
					 print(result)
					 new_results.add_result(estimator, data_set, metric, chunk, result[metric])
new_results.to_csv('lordhelpme.csv')
