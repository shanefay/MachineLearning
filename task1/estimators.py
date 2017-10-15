from sklearn import linear_model, svm, tree

class Estimators:
	regression = {
		'Linear Regression': linear_model.LinearRegression(),
		'Linear Support Vector Machine': svm.LinearSVR()
	}

	classification = {
		'Logistic Regression': linear_model.LogisticRegression(),
		'Decision Tree Classifier': tree.DecisionTreeClassifier()
	}