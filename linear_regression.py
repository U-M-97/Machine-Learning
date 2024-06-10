import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LinearRegression:
    def __init__(self, learning_rate = 0.01, num_iterations = 1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, x, y):
        samples, features = x.shape
        self.weights = np.zeros(features)
        self.bias = 0

        for _ in range(self.num_iterations):
            y_predicted = np.dot(x, self.weights) + self.bias

            dw = (1 / samples) * np.dot(x.T, (y_predicted - y))
            db = (1 / samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Calculate and store the cost for visualization
            cost = self.mean_squared_error(y, y_predicted)
            self.cost_history.append(cost)

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def predict(self, x):
        return np.dot(x, self.weights) + self.bias

def main():

    # df = pd.read_json(r'./datasets/archive/HPI_master.json')
    # df = pd.get_dummies(df, columns=['period', 'hpi_flavor', 'hpi_type', 'frequency', 'level', 'place_name', 'place_id', 'yr', 'index_nsa', 'index_sa',])
    # print(df.iloc[1])
    # x = df.drop(columns=['index_nsa'])  
    # y = df['index_nsa'] 
   
    # Generate input array x with 10000 entries
    x = np.array([[i, i+1] for i in range(1, 10001)])

    # Generate target array y with a linear relationship
    # y = np.array([i+2 for i in range(1, 10001)])

    # Generate target array y with a non-linear relationship (quadratic)
    y = np.array([i**2 + 2 for i in range(1, 10001)])

    model = LinearRegression()
    model.fit(x, y)
    predict = model.predict(x)
    plt.scatter(x[:, 0], y, color='blue', label='Actual data')
    plt.plot(x[:, 0], predict, color='red', label='regression line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

main()
