import numpy as np
import pandas as pd

class Perceptron:
    def __init__(self, input_size, learning_rate=0.3, epochs=100):
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.epochs = epochs
        self.weights = np.random.uniform(-1, 1, input_size)
        self.bias = np.random.uniform(-1, 1)

    def activation(self, net):
        return np.where(net > 0, 1, 0)

    def predict(self, inputs):
        net = np.dot(inputs, self.weights) + self.bias
        return self.activation(net)

    def train(self, x, y):
        for epoch in range(self.epochs):
            total_error = 0
            for inputs, expected in zip(x, y):
                prediction = self.predict(inputs)
                error = expected - prediction
                total_error += np.abs(error)
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error
            print(epoch)
            if total_error == 0:
                break

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([1, 1, 1, 0])
p = Perceptron(input_size=2)
p.train(x, y)

for inputs in x:
    print(f"Giriş: {inputs} Çıktı:{p.predict(inputs)}")