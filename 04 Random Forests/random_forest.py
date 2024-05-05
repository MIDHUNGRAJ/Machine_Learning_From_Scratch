from decisiontree import DecisionTree
from collections import Counter
import numpy as np


class RandomForest():
    def __init__(self, max_depth=10, num_trees=10, min_samples_split=2):

        self.max_depth = max_depth
        self.num_trees = num_trees
        self.min_samples_split = min_samples_split

    def fit(self, x, y):
        self.trees = []
        for _ in range(self.num_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split)
            
            X_sample, y_sample = self.bootstrap_samples(x, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
         

    def bootstrap_samples(self, X, y):
        num_samples = X.shape[0]
        idx = np.random.choice(num_samples, num_samples, replace=True)
        return X[idx], y[idx]

    def most_common_label(self, y):
        common = Counter(y).most_common(1)[0][0]
        return common

    def predict(self, x):
        predictions = np.array([tree.predict(x) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self.most_common_label(pred) for pred in tree_preds])
        return predictions
    
    

if __name__ == "__main__":
    from random import random
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import numpy as np
    from random_forest import RandomForest

    data = datasets.load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=6
    )

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    clf = RandomForest(num_trees=20)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    acc =  accuracy(y_test, predictions)
    print(acc)



