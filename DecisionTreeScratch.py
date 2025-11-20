import numpy as np

# ------------------------------
# Helper functions
# ------------------------------
def gini(y):
    """Compute the Gini impurity of a label array y."""
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)


def split_dataset(X, y, feature_index, threshold):
    """Split dataset into left/right branches based on threshold."""
    left_mask = X[:, feature_index] <= threshold
    right_mask = ~left_mask
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]


def best_split(X, y):
    """Find the best feature and threshold to split on."""
    best_gini = 1.0
    best_feature, best_threshold = None, None
    n_features = X.shape[1]
    for feature_index in range(n_features):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            X_left, y_left, X_right, y_right = split_dataset(X, y, feature_index, threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            g = (len(y_left)/len(y))*gini(y_left) + (len(y_right)/len(y))*gini(y_right)
            if g < best_gini:
                best_gini = g
                best_feature = feature_index
                best_threshold = threshold
    return best_feature, best_threshold


class DecisionTreeNode:
    """A simple decision tree node."""
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None


class DecisionTreeClassifierScratch:
    """Decision Tree Classifier from scratch using Gini impurity."""
    def __init__(self, max_depth=50, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_classes_ = None
        self.n_features_ = None
        self.tree_ = None

    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = DecisionTreeNode(
            gini=gini(y),
            num_samples=len(y),
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class
        )
        if depth < self.max_depth and len(y) >= self.min_samples_split and node.gini > 0:
            feature_index, threshold = best_split(X, y)
            if feature_index is not None:
                X_left, y_left, X_right, y_right = split_dataset(X, y, feature_index, threshold)
                node.feature_index = feature_index
                node.threshold = threshold
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _predict_one(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

    def predict(self, X):
        return np.array([self._predict_one(inputs) for inputs in X])

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
