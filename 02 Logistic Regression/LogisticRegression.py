import numpy as np

class LogisticReg:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        # Initialize learning rate and number of iterations
        self.lr = learning_rate
        self.max_iter = max_iter
        self.cost_history = []  # Track cost at each iteration (for analysis)

    def sigmoid_fn(self, z):
        # Sigmoid activation function to squash output between 0 and 1
        return 1 / (1 + np.exp(-z))

    def cost_fun(self, X, y):
        # Computes binary cross-entropy cost
        m = len(y)
        predictions = self.sigmoid_fn(np.dot(X, self.w))
        epsilon = 1e-9  # Avoid log(0) by adding small epsilon
        cost = (-1 / m) * np.sum(
            y * np.log(predictions + epsilon) +
            (1 - y) * np.log(1 - predictions + epsilon)
        )
        return cost

    def fit(self, X, y):
        # Train logistic regression model using gradient descent
        m, n = X.shape
        
        # Add bias term (column of 1s) to feature matrix X
        X = np.c_[np.ones((m, 1)), X]
        
        # Initialize weights (including bias weight) to zeros
        self.w = np.zeros(X.shape[1])

        # Gradient descent loop
        for _ in range(self.max_iter):
            # Calculate predictions using current weights
            predictions = self.sigmoid_fn(np.dot(X, self.w))

            # Compute gradient of cost w.r.t. weights
            gradient = (1 / m) * np.dot(X.T, (predictions - y))

            # Update weights in direction of negative gradient
            self.w -= self.lr * gradient

            # Compute and store cost after the update
            cost = self.cost_fun(X, y)
            self.cost_history.append(cost)

    def predict(self, X):
        # Predict class labels for input data X
        m = X.shape[0]

        # Add bias term (same as done during training)
        X = np.c_[np.ones((m, 1)), X]

        # Compute predicted probabilities using sigmoid
        predictions = self.sigmoid_fn(np.dot(X, self.w))

        # Apply threshold to convert probabilities to class labels
        return (predictions >= 0.5).astype(int)


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
data = load_iris()
X = data.data
y = data.target

# Binary classification (class 0 vs 1)
X = X[y != 2]
y = y[y != 2]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticReg(learning_rate=0.1, max_iter=1000)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = np.mean(predictions == y_test)
print("Accuracy:", accuracy)
