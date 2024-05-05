import numpy as np

class KNNRegressor:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self.pred_(data) for data in X_test]
        return predictions

    def pred_(self, data):
        distances = [self.euclidean_distance(data, x) for x in self.X_train]
        nearest_indices = np.argsort(distances)[:self.k]
        nearest_labels = [self.y_train[i] for i in nearest_indices]
        predicted_value = np.mean(nearest_labels)
        return predicted_value
    
    def euclidean_distance(self, x1, x2):
        distance = np.sqrt(np.sum((x1-x2)**2))
        return distance

if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    iris = datasets.load_iris()

    data = iris.data
    target = iris.target

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

    knn_classifier = KNNRegressor(k=3)
    knn_classifier.fit(X_train, y_train)
    predictions = knn_classifier.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"MSE: {mse}")

