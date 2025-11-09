from sklearn.naive_bayes import GaussianNB


#Trains and evaluates a Gaussian Naive Bayes model using sklearn

#create and train the model
def train_and_predict(X_train, y_train, X_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    return y_predict

