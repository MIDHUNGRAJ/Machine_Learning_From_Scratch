from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

iris = datasets.load_iris()

data = iris.data
target = iris.target

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

knn_regressor = KNeighborsRegressor(n_neighbors=3) 
knn_regressor.fit(X_train, y_train)
predictions = knn_regressor.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"MSE: {mse}")

