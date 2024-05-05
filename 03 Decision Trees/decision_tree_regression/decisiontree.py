from collections import Counter
import numpy as np
from utils import TreeNode

class DecisionTree():
    def __init__(self, max_depth=4, criterion="mse"):
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_leaf = 1

    def _build_tree(self, data, current_depth=0):
        """
        Recursively builds the decision tree.

        Args:
            data: The dataset for which the tree is being built.
            current_depth: The current depth of the tree.

        Returns:
            A TreeNode object representing the root of the built subtree.
        """

        min_samples, min_features = data.shape

        if current_depth <= self.max_depth and min_samples >= self.min_samples_leaf:

            left_split, right_split, best_threshold, best_feature, ctrn = self.find_best_split(data)
            if ctrn > 0:
                current_depth += 1
                left_data = self._build_tree(left_split, current_depth)
                right_data = self._build_tree(right_split, current_depth)
                return TreeNode(data_left=left_data, 
                                data_right=right_data,
                                best_feature=best_feature,
                                best_threshold=best_threshold,
                                var_red=ctrn)
            
        prob_label = Counter(data[:, -1]).most_common(1)[0][0]
        
        return TreeNode(prob=prob_label)
    
    def fit(self, x_train, y_train):
        """
        Fits the decision tree model to the training data.

        Args:
            x_train: The input features of the training data.
            y_train: The target labels of the training data.
        """
        train_data = np.concatenate((x_train, y_train.reshape(-1, 1)), axis=1)
        self.root = self._build_tree(train_data)

    def find_best_split(self, data):
        """
        Finds the best split for the given dataset.

        Args:
            data: The dataset to find the best split for.

        Returns:
            left_split: The left split of the dataset.
            right_split: The right split of the dataset.
            best_threshold: The threshold value for the best split.
            best_feature: The index of the best feature for splitting.
            best_criterion_value: The criterion value for the best split.
        """
        feature_idx = list(range(data.shape[1]-1))

        best_criterion_value = np.inf if self.criterion == 'mse' else -np.inf

        for idx in feature_idx:
            threshold = np.percentile(data[:, idx], q=np.arange(25, 100, 25))
            for tsd in threshold:
                left_g1, right_g2 = self.split_data(data, idx, tsd)
                
                if self.criterion == "mse":
                    criterion_value = self._calculate_mse(left_g1[:, -1]) + self._calculate_mse(right_g2[:, -1])
                    if criterion_value < best_criterion_value:
                        left_split = left_g1
                        right_split = right_g2
                        best_feature = idx
                        best_threshold = tsd
                        best_criterion_value = criterion_value

                elif self.criterion == "var":
                    criterion_value = self._calculate_variance_reduction(data[:, -1], left_g1[:, -1], right_g2[:, -1])
                    if criterion_value > best_criterion_value:
                        left_split = left_g1
                        right_split = right_g2
                        best_feature = idx
                        best_threshold = tsd
                        best_criterion_value = criterion_value

        return left_split, right_split, best_threshold, best_feature, best_criterion_value
        

    def split_data(self, data, idx, tsd):
        """
        Splits the dataset into left and right based on the threshold.

        Args:
            data: The dataset to split.
            idx: The index of the feature to split on.
            tsd: The threshold value for splitting.

        Returns:
            main_data_left: The left split of the dataset.
            main_data_right: The right split of the dataset.
        """
        main_data_condition = data[:, idx] < tsd
        main_data_left = data[main_data_condition]
        main_data_right = data[~main_data_condition]

        return main_data_left, main_data_right
    
    def _calculate_mse(self, y):
        """
        Calculates the mean squared error (MSE) for the given target values.

        Args:
            y: The target values.

        Returns:
            The mean squared error.
        """
        return np.mean((y - np.mean(y)) ** 2)
    
    def _calculate_variance_reduction(self, node, data_left, data_right):
        """
        Calculates the variance reduction.

        Args:
            node: The parent node.
            data_left: The left split of the data.
            data_right: The right split of the data.

        Returns:
            The variance reduction.
        """
        g_left = len(data_left) / len(node)
        g_right = len(data_right) / len(node)
        return np.var(node) - (g_left * np.var(data_left) + g_right * np.var(data_right))

    def print_tree(self, tree=None, level=0):
        """
        Prints the tree structure recursively.

        Args:
            tree: The root node of the tree.
            level: The current level of the tree.
        """
        if tree is None:
            tree = self.root

        if tree.pred_prob is not None:
            print(tree.pred_prob)
                
        else:
            print(f"X_{tree.feature_idx} <= {tree.threshold} var_reduction: {tree.var_red}")
            print(f"{'    ' * 2 * level}left:", end="")
            self.print_tree(tree.left, level + 1)
            print(f"{'    ' * 2 * level}right:", end="")
            self.print_tree(tree.right, level + 1)


    def _predict(self, dataset):
        """
        Predicts the target label for a single data point.

        Args:
            dataset: The input features.

        Returns:
            The predicted target label.
        """
        node = self.root

        while node.left is not None or node.right is not None:
            if dataset[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node.pred_prob

    def predict(self, data):
        """
        Predicts the target labels for multiple data points.

        Args:
            data: The input features.

        Returns:
            The predicted target labels.
        """
        return np.array([self._predict(sd) for sd in data])
    
    
if __name__ == "__main__":
    """
    This method will help you understand how this model works.
    """
    from utils import X_train, y_train, X_test, y_test, root_mean_squared_error
    test = DecisionTree()
    test.fit(X_train, y_train)
    test.print_tree()
    y_pred = test.predict(X_test)
    print(f"\nRMSE: {root_mean_squared_error(y_test, y_pred)}")

