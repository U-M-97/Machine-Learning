import numpy as np

class NaiveBayes:
    
    def fit(self, x, y):
        samples, features = x.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes, features), dtype=np.float64)
        self.var = np.zeros((n_classes, features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for i, c in enumerate(self.classes):
            x_c = x[y == c]
            self.mean[i, :] = x_c.mean(axis=0)
            self.var[i, :] = x_c.var(axis=0)
            self.priors[i] = x_c.shape[0] / float(samples)

    def predict(self, x):
        y_pred = [self._predict(x) for x in x]
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors = []

        for i, c in enumerate(self.classes):
            prior = np.log(self.priors[i])
            posterior = np.sum(np.log(self._pdf(i, x)))
            posterior = posterior + prior
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_i, x): 
        return np.exp(-((x - self.mean[class_i]) ** 2) / (2 * (self.var[class_i] ** 2 ))) / np.sqrt(2 * np.pi * (self.var[class_i] ** 2))
   

if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=123
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)

    print("Naive Bayes classification accuracy", accuracy(y_test, predictions))