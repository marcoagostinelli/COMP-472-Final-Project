import numpy as np

class GaussianNaiveBayes:
    def __init__(self):
        # Initialize the Gaussian Naive Bayes classifier
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}

    def fit(self,X,y):
        """
        Train the model by estimating mean, variance, and class priors from training data.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y) # Get unique class labels
        n_classes = len(self.classes)

        # Initialize mean, variance, and prior for each class
        self.mean = np.zeros((n_classes, n_features)) # mean for each class and feature
        self.var = np.zeros((n_classes, n_features)) # variance for each class and feature
        self.priors = np.zeros(n_classes) # prior probability for each class

        # Calculate mean, variance, and prior for each class
        for idx, c in enumerate(self.classes):
            #get all samples belonging to class c
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / n_samples



    def _calculate_likelihood(self, class_idx, x):
        """
        Calculate the Gaussian probability density function
        
        Args:
            class_idx: Index of the class
            x: Feature vector

        Returns:
            Log-likelihood value.
        """
        mean = self.mean[class_idx]
        var = self.var[class_idx]

        #Add small value to variance to avoid division by zero
        epsilon = 1e-9

        # Using Gaussian log-probability for each feature
        numerator = -((x - mean) ** 2) / (2 * (var + epsilon))
        denominator = -0.5 * np.log(2 * np.pi * (var + epsilon))

        #combine log-probabilities of all features
        log_likelihood = np.sum(numerator + denominator)
        return log_likelihood

    def _calculate_posterior(self, x):
        """
        Calculate posterior probability for each class given a sample x.
        
        Args:
            x: Feature vector

        Returns:
            Array of posterior probabilities for each class.
        """
        posteriors = []
        for idx, c in enumerate(self.classes):
            #Prior: log(P(class))
            prior = np.log(self.priors[idx])

            #Likelihood: log(P(features | class))
            likelihood = self._calculate_likelihood(idx, x)

            #Posterior: log(P(class | features)) = log(P(features | class)) + log(P(class))
            posterior = prior + likelihood
            posteriors.append(posterior)
        return posteriors
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Args:
            X: Test features (n_samples, n_features)

        Returns:
            Array of predicted class labels.
        """
        predictions = []

        for x in X:
            posteriors = self._calculate_posterior(x)

            #Select class with highest posterior probability
            predicted_class = self.classes[np.argmax(posteriors)]
            predictions.append(predicted_class)
        return np.array(predictions)
