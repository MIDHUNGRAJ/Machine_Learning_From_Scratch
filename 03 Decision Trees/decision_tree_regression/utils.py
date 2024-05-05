from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from collections import Counter
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.2, random_state = 6)

class TreeNode():
    def __init__(self, data_left=None, data_right=None, best_feature=None, best_threshold=None, var_red=None, prob=None):
        self.left = data_left
        self.right = data_right
        self.feature_idx = best_feature
        self.threshold = best_threshold
        self.var_red = var_red
        self.pred_prob = prob
    
