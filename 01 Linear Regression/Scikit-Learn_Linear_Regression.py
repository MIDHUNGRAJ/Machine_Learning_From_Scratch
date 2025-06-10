from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor
import numpy as np

# Create synthetic regression data
X, y = make_regression(n_samples=200, n_features=1, noise=10, random_state=42)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Normal Equation (Closed Form) ---
normal_model = LinearRegression()
normal_model.fit(X_train, y_train)
y_pred_normal = normal_model.predict(X_test)
mse_normal = mean_squared_error(y_test, y_pred_normal)
print("MSE (Normal Equation):", mse_normal)

# --- Gradient Descent (SGD) ---
sgd_model = SGDRegressor(max_iter=1000, learning_rate='invscaling', eta0=0.01, random_state=42)
sgd_model.fit(X_train, y_train)
y_pred_sgd = sgd_model.predict(X_test)
mse_sgd = mean_squared_error(y_test, y_pred_sgd)
print("MSE (Gradient Descent):", mse_sgd)

# --- Visualization ---
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')

# Sorting X_test for smooth line plotting
sorted_indices = np.argsort(X_test[:, 0])
X_test_sorted = X_test[sorted_indices]

# Plot predictions
plt.plot(X_test_sorted, y_pred_normal[sorted_indices], color='green', label='Normal Equation')
plt.plot(X_test_sorted, y_pred_sgd[sorted_indices], color='red', linestyle='--', label='Gradient Descent (SGD)')

plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression: Normal Equation vs Gradient Descent')
plt.legend()
plt.grid(True)
plt.show()
