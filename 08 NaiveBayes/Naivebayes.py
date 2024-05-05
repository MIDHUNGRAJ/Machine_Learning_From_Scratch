import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.mean = np.zeros((self.n_classes, X.shape[1]))
        self.std = np.zeros((self.n_classes, X.shape[1]))
        self.priors = np.zeros(self.n_classes)

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[i, :] = X_c.mean(axis=0)
            self.std[i, :] = X_c.std(axis=0)
            self.priors[i] = len(X_c) / len(X)

    def _gaussian_pdf(self, X, mean, std):
        exponent = np.exp(-((X - mean) ** 2) / (2 * (std ** 2)))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

    def predict(self, X):
        posteriors = []

        for i, c in enumerate(self.classes):
            likelihood = np.prod(self._gaussian_pdf(X, self.mean[i, :], self.std[i, :]), axis=1)
            posterior = likelihood * self.priors[i]
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors, axis=0)]


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    from sklearn.metrics import accuracy_score

    X, y = datasets.load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    nb = NaiveBayes()
    nb.fit(X_train, y_train)

    print(accuracy_score(y_test, nb.predict(X_test)))
    

