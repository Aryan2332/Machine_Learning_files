# The following code is the build of most basic neural network code but the neural 
# network in this file has the custom layer and neurons architecture design choosing ability while running the program .
# Just provide the name of file at the place of "FILE_NAME" , file should be of .csv pattern ONLY 
# and exits in the same folder as the code . 
# The dataset file here can be approximately any data file having a column of "label" and
# others are feature of the data , point to be noted is that all values should we of "int" type only .
# I have tested this program on pixel based feature extraction with different dataset , its working fine .
# -------------------------------------------------------------------------------
# KINDLY REPORT FOR ANY ERRORS AND BUGS.          ---THANK YOU😊
# ------------------------------------------------------------------------------

import numpy as np
import pandas as pd

def load_dt_st():
    dt = pd.read_csv("FILE_NAME.csv")
    y = dt['label'].values
    dt = dt.drop(columns=['label']).values
    print(f"Following dataset has : {len(dt)} examples")
    x = int(input("Enter the number of training examples you want of all : "))
    x_trn1 = dt[:x] / 255.0  
    y_trn1 = y[:x]
    x_tst1 = dt[x:] / 255.0  
    y_tst1 = y[x:]
    print(f"Test data examples  {len(y_tst1)} and train {len(y_trn1)} examples")
    return x_trn1, y_trn1, x_tst1, y_tst1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers
        self.weights = []
        self.biases = []
        self.learning_rate = learning_rate
        for i in range(len(self.layers) - 1):
            weight = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2. / self.layers[i])
            bias = np.zeros((1, self.layers[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def forward(self, x):
        self.activations = [x]
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            x = sigmoid(np.dot(x, weight) + bias)
            self.activations.append(x)
        output = np.dot(x, self.weights[-1]) + self.biases[-1]
        output = softmax(output)
        self.activations.append(output)
        return output

    def backward(self, x, y, output):
        output_error = y - output
        output_delta = output_error
        self.weights[-1] += self.learning_rate * np.dot(self.activations[-2].T, output_delta)
        self.biases[-1] += self.learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        for i in range(2, len(self.layers)):
            layer_error = np.dot(output_delta, self.weights[-i + 1].T)
            layer_delta = layer_error * sigmoid_derivative(self.activations[-i])
            self.weights[-i] += self.learning_rate * np.dot(self.activations[-i - 1].T, layer_delta)
            self.biases[-i] += self.learning_rate * np.sum(layer_delta, axis=0, keepdims=True)
            output_delta = layer_delta

    def train(self, x, y, epochs, batch_size):
        for epoch in range(epochs):
            for i in range(0, x.shape[0], batch_size):
                x_batch = x[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                output = self.forward(x_batch)
                self.backward(x_batch, y_batch, output)
            if epoch % 20 == 0 or epoch == epochs - 1:
                predictions = self.predict(x)
                accuracy = np.mean(predictions == np.argmax(y, axis=1))
                print(f"Epoch {epoch + 1}/{epochs}, Training accuracy: {accuracy * 100:.2f}%")

    def predict(self, x):
        output = self.forward(x)
        return np.argmax(output, axis=1)


def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_dt_st()
    uq_el = set(y_train)
    ot_cls = len(uq_el)
    print("THE NUMBER OF DIFFERENT CLASSES FOLLOWING DATASET HAS : ", ot_cls)
    y_encoded = one_hot_encode(y_train, ot_cls)
    num_layers = int(input("Enter number of hidden layers: ")) + 2
    neurons_per_layer = list(map(int, input("Enter number of neurons in each hidden layer (space-separated): ").split()))
    layers = [X_train.shape[1]] + neurons_per_layer + [ot_cls]

    # ANN object creation
    nn = NeuralNetwork(layers, learning_rate=0.01)
    pochs = int(input("Enter the number of epochs you want : "))
    nn.train(X_train, y_encoded, epochs=pochs, batch_size=32)

    # Make predictions on training data
    predictions_train = nn.predict(X_train)
    accuracy_train = np.mean(predictions_train == y_train)
    print(f"Final Training accuracy: {accuracy_train * 100:.2f}%")

    # Make predictions on test data
    y_encoded_test = one_hot_encode(y_test, ot_cls)
    predictions_test = nn.predict(X_test)
    accuracy_test = np.mean(predictions_test == y_test)
    print("\n---------\n")
    print(f"Test accuracy: {accuracy_test * 100:.2f}%")
