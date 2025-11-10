import numpy as np
from GaussianNaiveBayes import GaussianNaiveBayes
from SklearnNaiveBayes import train_and_predict
from evaluateNaiveBayes import evaluate

# load the dataset
X_train = np.load('./processed_data/X_train_pca.npy')
y_train = np.load('./processed_data/y_train.npy')
X_test = np.load('./processed_data/X_test_pca.npy')
y_test = np.load('./processed_data/y_test.npy')

#run our Gaussian Naive Bayes implementation
gnb = GaussianNaiveBayes()
gnb.fit(X_train, y_train)
y_predict_custom = gnb.predict(X_test)
customResults = evaluate(y_test, y_predict_custom)

#run sklearn's Gaussian Naive Bayes implementation
y_predict_sklearn = train_and_predict(X_train, y_train, X_test)
sklearnResults = evaluate(y_test, y_predict_sklearn)
