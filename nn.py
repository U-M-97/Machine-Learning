import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import time

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    return np.exp(x) / sum(np.exp(x))

def initialize_parameters(input_layer_size, hidden_layer_size_1, hidden_layer_size_2, output_layer_size):
    w1 = np.random.rand(hidden_layer_size_1, input_layer_size) - 0.5
    b1 = np.random.rand(hidden_layer_size_1, 1) - 0.5
    w2 = np.random.rand(hidden_layer_size_2, hidden_layer_size_1) - 0.5
    b2 = np.random.rand(hidden_layer_size_2, 1) - 0.5
    w3 = np.random.rand(output_layer_size, hidden_layer_size_2) - 0.5
    b3 = np.random.rand(output_layer_size, 1) - 0.5
    return w1, b1, w2, b2, w3, b3

def forward_propagation(x, w1, b1, w2, b2, w3, b3):
    z1 = np.dot(w1, x.T) + b1
    a1 = relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = relu(z2)
    z3 = np.dot(w3, a2) + b3
    a3 = softmax(z3)
    return z1, a1, z2, a2, z3, a3

def loss(y, a3):
    m = y.shape[0]
    loss_value = np.sum(((y.T - a3) ** 2)) / m
    return loss_value

def backward_propagate(x, y, z1, a1, z2, a2, a3, w2, w3):
    m = x.shape[0]

    dz3 = a3 - y.T
    dw3 = np.dot(dz3, a2.T) / m
    db3 = np.sum(dz3) / m

    da2 = np.dot(w3.T, dz3)
    dz2 = da2 * relu_derivative(z2)
    dw2 = np.dot(dz2, a1.T) / m
    db2 = np.sum(dz2) / m

    da1 = np.dot(w2.T, dz2)
    dz1 = da1 * relu_derivative(z1)
    dw1 = np.dot(dz1, x) / m
    db1 = np.sum(dz1) / m

    return dw1, db1, dw2, db2, dw3, db3

def update_parameters(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, learning_rate):
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2
    w3 -= learning_rate * dw3
    b3 -= learning_rate * db3

    return w1, b1, w2, b2, w3, b3

def train(x, y, input_layer_size, hidden_layer_size_1, hidden_layer_size_2, output_layer_size, learning_rate, iterations):
    w1, b1, w2, b2, w3, b3 = initialize_parameters(input_layer_size, hidden_layer_size_1, hidden_layer_size_2, output_layer_size)
    
    for i in range(iterations):
        z1, a1, z2, a2, z3, a3 = forward_propagation(x, w1, b1, w2, b2, w3, b3)
        loss_value = loss(y, a3)
        dw1, db1, dw2, db2, dw3, db3 = backward_propagate(x, y, z1, a1, z2, a2, a3, w2, w3)
        w1, b1, w2, b2, w3, b3 = update_parameters(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, learning_rate)

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss_value}")

    return w1, b1, w2, b2, w3, b3

def main():
    mnist = fetch_openml('mnist_784', version=1)
    x, y = mnist['data'], mnist['target']

    x = np.array(x, dtype='float32')
    y = np.array(y, dtype='int32')
    x /= 255.0

    num_classes = 10
    y_onehot = np.zeros((y.shape[0], num_classes))
    for i in range(len(y)):
        y_onehot[i, y[i]] = 1

    x_train, x_test, y_train, y_test = train_test_split(x, y_onehot, test_size=0.2, random_state=42)
    
    input_layer_size = x_train.shape[1]
    hidden_layer_size_1 = 16
    hidden_layer_size_2 = 16
    output_layer_size = num_classes
    learning_rate = 0.01
    iterations = 20000

    # start = time.time()
    # w1, b1, w2, b2, w3, b3 = train(x_train, y_train, input_layer_size, hidden_layer_size_1, hidden_layer_size_2, output_layer_size, learning_rate, iterations)
    # end = time.time()
    # print("total time to train model", end - start, "seconds")
    # np.savez("mnist_params.npz", w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)

    # params = np.load('mnist_params.npz')
    # _, _, _, _, _, y_predict = forward_propagation(x_test[3000].reshape(1, -1), params['w1'], params['b1'], params['w2'], params['b2'], params['w3'], params['b3'])
    # print(np.argmax(y_predict))
    # plt.figure(figsize=(10, 5))

    # plt.subplot(1, 2, 1)
    # plt.imshow(x_test[3000].reshape(28, 28), cmap='gray')
    # plt.title('Real Image')

    # plt.subplot(1, 2, 2)
    # plt.text(0.5, 0.5, str(np.argmax(y_predict)), fontsize=120, ha='center')
    # plt.title('Predicted Number')
    # plt.axis('off')

    # plt.tight_layout()
    # plt.show()
    # accuracy = np.mean(np.argmax(y_predict.T, axis=1) == np.argmax(y_test, axis=1)) * 100
    # print("Accuracy = ", accuracy, " % ")

    # img = cv2.imread("images/7.jpg", cv2.IMREAD_GRAYSCALE)
    # plt.imshow(img, cmap='gray')
    # plt.axis('off') 
    # plt.show()
    # img = cv2.resize(img, (28, 28))
    # plt.imshow(img, cmap='gray')
    # plt.axis('off')  
    # plt.show()
    # img = img.flatten()
    # img = np.array(img, dtype='float32')
    # img /= 255.0
    # img = img.reshape(1, -1)

    # _, _, _, _, _, y_predict = forward_propagation(img, params['w1'], params['b1'], params['w2'], params['b2'], params['w3'], params['b3'])
    # print(y_predict)
    # print(np.argmax(y_predict))
   
main()
