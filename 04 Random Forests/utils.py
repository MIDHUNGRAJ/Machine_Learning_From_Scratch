from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score


iris = load_breast_cancer()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.2, random_state = 6)

class TreeNode():
    def __init__(self, data_left=None, data_right=None, best_feature=None, best_threshold=None, information_gain=None, prob=None):
        self.left = data_left
        self.right = data_right
        self.feature_idx = best_feature
        self.threshold = best_threshold
        self.info_gain = information_gain
        self.pred_prob = prob
    
