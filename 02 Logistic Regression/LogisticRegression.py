import numpy as np

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function
def compute_cost(X, y, weights):
    m = len(y)
    z = np.dot(X, weights)
    predictions = sigmoid(z)
    cost = (-1 / m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return cost

# Gradient descent
def gradient_descent(X, y, weights, learning_rate, iterations):
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        z = np.dot(X, weights)
        predictions = sigmoid(z)
        gradient = np.dot(X.T, (predictions - y)) / m
        weights -= learning_rate * gradient

        cost = compute_cost(X, y, weights)
        cost_history.append(cost)

    return weights, cost_history

# Logistic regression model
def logistic_regression(X, y, learning_rate=0.01, iterations=1000):
    X = np.c_[np.ones(X.shape[0]), X]  # Add bias term
    weights = np.zeros(X.shape[1])
    weights, cost_history = gradient_descent(X, y, weights, learning_rate, iterations)
    return weights, cost_history

# Predict function
def predict(X, weights):
    X = np.c_[np.ones(X.shape[0]), X]  # Add bias term
    probabilities = sigmoid(np.dot(X, weights))
    return [1 if p >= 0.5 else 0 for p in probabilities]

# Example usage:
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    # Load Iris dataset
    data = load_iris()
    X = data.data  # Features
    y = data.target  # Labels

    # Select two classes for binary classification (class 0 and 1)
    X = X[y != 2]
    y = y[y != 2]

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train logistic regression model
    weights, cost_history = logistic_regression(X_train, y_train, learning_rate=0.1, iterations=1000)

    print("Trained weights:", weights)
    print("Cost history:", cost_history[:5])  # Print first 5 costs

    # Predictions
    predictions = predict(X_test, weights)
    print("Predictions:", predictions)
