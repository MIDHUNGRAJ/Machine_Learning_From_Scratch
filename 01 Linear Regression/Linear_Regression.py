import numpy as np

class LinearRegression():
    def __init__(self, learning_rate, max_iter):
        self.lr = learning_rate
        self.max_iter = max_iter

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        self.weights = np.random.rand(n_features)
        self.bias = 0

        for _ in range(self.max_iter):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    



if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    X, y = make_regression(n_samples=200, n_features=1, noise=10, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = LinearRegression(learning_rate=0.01, max_iter=1000)
    reg.fit(X=X_train, y=y_train)
    y_pred = reg.predict(X_test)

    print(f"{mean_squared_error(y_test, y_pred):.3f}")

    

