import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm_notebook


class NeuralNet:

    def __init__(self, input_size, num_hidden_nodes, num_colors, learning_rate, labels):
        """
        Initialize neural network with basic parameters
        """

        self.W1 = np.random.randn(num_hidden_nodes, input_size)
        self.b1 = np.zeros((num_hidden_nodes, 1))
        self.W2 = np.random.randn(num_colors, num_hidden_nodes)
        self.b2 = np.zeros((num_colors, 1))
        self.learning_rate = learning_rate
        self.label_encoder = LabelEncoder().fit(labels)
        self.num_colors = num_colors

    def fit(self, X_train, Y_train, epochs=100):
        """
        Train classification model
        """

        for i in tqdm_notebook(range(epochs)):
            for X, Y in zip(X_train, Y_train):

                m = Y.shape[0]
                Y = Y.flatten()
                Y = self.label_encoder.transform(Y)
                Y = Y.reshape(1, len(Y))
                Y = np.eye(self.num_colors)[Y.astype('int32')]
                Y = Y.T.reshape(self.num_colors, m)

                Z1 = np.matmul(self.W1, X) + self.b1
                A1 = self.sigmoid(Z1)
                Z2 = np.matmul(self.W2, A1) + self.b2
                A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)

                cost = self.compute_multiclass_loss(Y, A2)

                dZ2 = A2-Y
                dW2 = (1./m) * np.matmul(dZ2, A1.T)
                db2 = (1./m) * np.sum(dZ2, axis=1, keepdims=True)

                dA1 = np.matmul(self.W2.T, dZ2)
                dZ1 = dA1 * self.sigmoid(Z1) * (1 - self.sigmoid(Z1))
                dW1 = (1./m) * np.matmul(dZ1, X.T)
                db1 = (1./m) * np.sum(dZ1, axis=1, keepdims=True)

                self.W2 = self.W2 - self.learning_rate * dW2
                self.b2 = self.b2 - self.learning_rate * db2
                self.W1 = self.W1 - self.learning_rate * dW1
                self.b1 = self.b1 - self.learning_rate * db1

            if (i % 10 == 0):
                print("Epoch", i, "cost: ", cost)

        print("Final cost:", cost)

    def predict(self, x_image, y_image):

        Z1 = np.matmul(self.W1, x_image) + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = np.matmul(self.W2, A1) + self.b2
        A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)

        m = y_image.shape[0]
        y_image = y_image.flatten()
        y_image = self.label_encoder.transform(y_image)
        y_image = y_image.reshape(1, len(y_image))
        y_image = np.eye(self.num_colors)[y_image.astype('int32')]
        y_image = y_image.T.reshape(self.num_colors, m)

        predictions = np.argmax(A2, axis=0)
        labels = np.argmax(y_image, axis=0)

        reconstruction = []
        for label in self.label_encoder.inverse_transform(predictions):
            reconstruction.append(
                np.array([float(x) for x in label.split(',')]))

        return reconstruction

    def sigmoid(self, z):
        """
        Perform Sigmoid Function
        """
        s = 1 / (1 + np.exp(-z))
        return s

    def compute_multiclass_loss(self, Y, Y_hat):
        """
        Calculate loss
        """

        L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
        m = Y.shape[1]
        L = -(1/m) * L_sum

        return L
