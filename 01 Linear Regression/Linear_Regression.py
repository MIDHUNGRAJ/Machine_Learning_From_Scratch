import numpy as np

class LinearRegression:
    def __init__(self, learning_rate, max_iter):
        # Initialize learning rate and maximum number of iterations
        self.lr = learning_rate
        self.max_iter = max_iter

    def fit(self, X, y):
        # Get the number of samples and features
        n_samples, n_features = X.shape

        # Initialize weights randomly and bias to 0
        self.weights = np.random.rand(n_features)
        self.bias = 0

        # Perform gradient descent
        for _ in range(self.max_iter):
            # Predict the output using current weights and bias
            y_pred = np.dot(X, self.weights) + self.bias

            # Calculate the gradients (partial derivatives)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))  # Gradient w.r.t. weights
            db = (1 / n_samples) * np.sum(y_pred - y)         # Gradient w.r.t. bias

            # Update weights and bias using the gradients
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        # Predict output using learned weights and bias
        return np.dot(X, self.weights) + self.bias

    
if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    # Create synthetic linear data with noise
    X, y = make_regression(n_samples=200, n_features=1, noise=10, random_state=42)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and train the model
    reg = LinearRegression(learning_rate=0.01, max_iter=1000)
    reg.fit(X=X_train, y=y_train)

    # Predict on test data
    y_pred = reg.predict(X_test)

    # Evaluate the model using Mean Squared Error
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.3f}")

    

