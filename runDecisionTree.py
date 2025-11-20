import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from SklearnDecisionTree import train_sklearn_tree
from DecisionTreeScratch import DecisionTreeClassifierScratch

# Load data
X_train = np.load("processed_data/X_train_pca.npy")
X_test = np.load("processed_data/X_test_pca.npy")
y_train = np.load("processed_data/Y_train.npy")
y_test = np.load("processed_data/Y_test.npy")

# --- Sklearn Version ---
print("=== Scikit-learn Decision Tree ===")
clf = train_sklearn_tree(X_train, y_train, max_depth=50)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Scratch Version ---
print("\n=== From Scratch Decision Tree ===")
tree = DecisionTreeClassifierScratch(max_depth=50)
tree.fit(X_train, y_train)
y_pred_scratch = tree.predict(X_test)
print("Accuracy:", np.mean(y_pred_scratch == y_test))
