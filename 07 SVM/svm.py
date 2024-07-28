import numpy as np

class SVM():
    def __init__(self, learning_rate=0.001, lambda_param=0.01, iter=1000):
        self.lr = learning_rate
        self.lm = lambda_param
        self.it = iter
        self.w = None
        self.b = None

    def fit(self, X, y):
        
        n_samples , n_features = X.shape
        y_ = np.where(y<=0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.it):
            for idx, x_data in enumerate(X):
                con_check = y_[idx] * (np.dot(x_data, self.w) - self.b) >= 1
                if con_check:
                    self.w -= self.lr * (2 * self.lm * self.w)
                else:
                    self.w -= self.lr * (2 * self.lm * self.w - np.dot(x_data, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, x):
        output = np.dot(x, self.w) - self.b
        y_out = np.sign(output)
        y_hat = np.where(y_out <= -1,0, 1)
        return y_hat



if __name__ == "__main__":

    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import matplotlib.pyplot as plt

    X, y = datasets.make_blobs(
        n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40
    )
    # y = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    model = SVM()
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    
    print("SVM classification accuracy: ", accuracy(y_test, prediction))


def plot_svm_decision_boundary(model, X, y):
    def calculate_hyperplane_y(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    

    plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)
    

    x0_min = np.amin(X[:, 0])
    x0_max = np.amax(X[:, 0])
    

    y0_hyperplane_min = calculate_hyperplane_y(x0_min, model.w, model.b, 0)
    y0_hyperplane_max = calculate_hyperplane_y(x0_max, model.w, model.b, 0)
    
    y0_margin_neg_min = calculate_hyperplane_y(x0_min, model.w, model.b, -1)
    y0_margin_neg_max = calculate_hyperplane_y(x0_max, model.w, model.b, -1)
    
    y0_margin_pos_min = calculate_hyperplane_y(x0_min, model.w, model.b, 1)
    y0_margin_pos_max = calculate_hyperplane_y(x0_max, model.w, model.b, 1)
    
    ax.plot([x0_min, x0_max], [y0_hyperplane_min, y0_hyperplane_max], "y--")

    ax.plot([x0_min, x0_max], [y0_margin_neg_min, y0_margin_neg_max], "k")
    ax.plot([x0_min, x0_max], [y0_margin_pos_min, y0_margin_pos_max], "k")
    
    y_min = np.amin(X[:, 1])
    y_max = np.amax(X[:, 1])
    ax.set_ylim([y_min - 3, y_max + 3])
    
    plt.show()

plot_svm_decision_boundary(model, X, y)

