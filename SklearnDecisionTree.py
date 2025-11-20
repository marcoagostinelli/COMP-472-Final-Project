from sklearn.tree import DecisionTreeClassifier

def train_sklearn_tree(X_train, y_train, max_depth=50, criterion="gini", random_state=42):
    """
    Trains a Decision Tree classifier using Scikit-learn.

    Parameters:
        X_train (ndarray): Training features
        y_train (ndarray): Training labels
        max_depth (int): Maximum depth of the tree
        criterion (str): Split quality criterion ('gini' or 'entropy')
        random_state (int): Random seed for reproducibility

    Returns:
        clf (DecisionTreeClassifier): Trained model
    """
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion=criterion,
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    return clf
