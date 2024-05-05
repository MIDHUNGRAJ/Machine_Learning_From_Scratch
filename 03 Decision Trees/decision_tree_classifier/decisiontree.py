from collections import Counter
import numpy as np
from utils import TreeNode


class DecisionTree():
    def __init__(self, max_depth=4, criterion='gini', min_samples_split=1):
        """Initialize the DecisionTree class with parameters."""
        self.max_depth = max_depth 
        self.criterion = criterion 
        self.min_information_gain = 0 
        self.min_samples_split = min_samples_split

    def build_tree(self, data, current_depth=0):
        """Recursive function to build the decision tree."""
        min_samples, min_features = data.shape
        if current_depth <= self.max_depth and min_samples > self.min_samples_split:
            """Check if current depth is less than or equal to the maximum depth."""
            
            left_split, right_split, best_threshold, best_feature, info_gain = self.find_best_split(data)
            if info_gain > self.min_information_gain:
                """If information gain is greater than 0, continue splitting."""
                
                current_depth += 1
                
                left_data = self.build_tree(left_split, current_depth)
                right_data = self.build_tree(right_split, current_depth)
                

                return TreeNode(data_left=left_data, 
                                data_right=right_data,
                                best_feature=best_feature,
                                best_threshold=best_threshold,
                                information_gain=info_gain)
            

        prob_label = Counter(data[:, -1]).most_common(1)[0][0]
        return TreeNode(prob=prob_label)
    
    def fit(self, x_train, y_train):
        """Fit the decision tree model to the training data."""
        train_data = np.concatenate((x_train, y_train.reshape(-1, 1)), axis=1)

        self.root = self.build_tree(train_data)

    def find_best_split(self, data):
        """Find the best split for the given data."""
        feature_idx = list(range(data.shape[1]-1)) 
        mx_info_gain = -1e10  
        
        for idx in feature_idx:
            threshold = np.percentile(data[:, idx], q=np.arange(25, 100, 25))  # Compute percentile thresholds
            for tsd in threshold:
                left_g1, right_g2 = self.split_data(data, idx, tsd)
                cr_info_gain = self.information_gain(data[:, -1], left_g1[:, -1], right_g2[:, -1], self.criterion)
                if cr_info_gain > mx_info_gain:
                    left_split = left_g1
                    right_split = right_g2
                    best_feature = idx
                    best_threshold = tsd
                    mx_info_gain = cr_info_gain
        
        return left_split, right_split, best_threshold, best_feature, mx_info_gain
    
    def split_data(self, data, idx, tsd):
        """Split the data into left and right based on a threshold."""
        main_data_condition = data[:, idx] < tsd  
        main_data_left = data[main_data_condition]  
        main_data_right = data[~main_data_condition]  
        return main_data_left, main_data_right

    def information_gain(self, y_node, g_left, g_right, ctrn="entropy"):
        """Calculate information gain for a split."""
        w_left = len(g_left) / len(y_node)  
        w_right = len(g_right) / len(y_node)  

        if ctrn == "gini":
 
            gain = self.gini(y_node) - (w_left*self.gini(g_left) + w_right*self.gini(g_right))
        else:

            gain = self.entropy(y_node) - (w_left*self.entropy(g_left) + w_right*self.entropy(g_right))
        
        return gain
    
    def gini(self, subset):
        """Calculate Gini impurity."""
        total_len = len(subset)  
        labels = np.unique(subset)
        clt = Counter(subset)  
        gini = 0
        for i in labels:
            p = clt[i] / total_len  
            gini += p**2

        return 1 - gini

    def entropy(self, subset):
        """Calculate entropy."""
        total_len = len(subset)  
        labels = np.unique(subset) 
        clt = Counter(subset)  
        entropy = 0
        for i in labels:
            p = clt[i] / total_len  
            entropy += -p * np.log2(p) 

        return entropy

    def print_tree(self, tree=None, level=0):
        """Print the decision tree."""
        if tree is None:
            tree = self.root

        if tree.pred_prob is not None:
            print(tree.pred_prob)
                
        else:
            print(f"X_{tree.feature_idx} <= {tree.threshold} gain: {tree.info_gain}")
            print(f"{'    ' * 2 * level}left:", end="")
            self.print_tree(tree.left, level + 1)
            print(f"{'    ' * 2 * level}right:", end="")
            self.print_tree(tree.right, level + 1)

    def _predict(self, dataset):
        """Predict the label for a single data point."""
        node = self.root

        while node.left is not None or node.right is not None:
            if dataset[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node.pred_prob

    def predict(self, data):
        """Predict labels for multiple data points."""
        return np.array([self._predict(sd) for sd in data])

if __name__ == "__main__":
    """
    This method will help you understand how this model works.
    """
    from utils import X_train, y_train, X_test, y_test, accuracy_score
    test = DecisionTree(criterion="entropy")
    test.fit(X_train, y_train)
    test.print_tree()
    y_pred = test.predict(X_test)
    print(f"\nAccuracy Score: {accuracy_score(y_test, y_pred)}")

