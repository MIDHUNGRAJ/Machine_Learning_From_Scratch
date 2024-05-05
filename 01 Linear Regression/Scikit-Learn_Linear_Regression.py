from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X, y = make_regression(n_samples=200, n_features=1, noise=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(mean_squared_error(y_test, y_pred))

plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Dataset')
plt.show()

