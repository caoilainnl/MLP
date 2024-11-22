import numpy as np


class MLP:

    def __init__(self, sizes, learning_rate=0.01, momentum=0.9, beta=1.0):
        """
        Initialize the MLP.

        Parameters:
        - sizes: List containing the number of neurons in each layer. 
          Example: [784, 128, 64, 10] for an input layer with 784 features, two hidden layers 
          with 128 and 64 neurons respectively, and an output layer with 10 neurons.
        - learning_rate: Learning rate for gradient descent.
        - momentum: Momentum coefficient for gradient descent.
        - beta: Scaling factor for activations (default is 1.0).
        """
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.beta = beta

        # Initialize weights and biases
        self.weights = [np.random.randn(sizes[i], sizes[i+1]) * 0.01 for i in range(self.num_layers - 1)]
        self.biases = [np.zeros((1, sizes[i+1])) for i in range(self.num_layers - 1)]

        # Initialize velocity terms for momentum
        self.velocities_w = [np.zeros_like(w) for w in self.weights]
        self.velocities_b = [np.zeros_like(b) for b in self.biases]

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forwardPass(self, X):
        """

        Parameters:
        - X: Input data.

        Returns:
        - activations: List of activations for each layer.
        """
        activations = [X]
        for i in range(self.num_layers - 2):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self.sigmoid(self.beta * z)
            activations.append(a)

        # Output layer with softmax activation
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        a = self.softmax(z)
        activations.append(a)

        return activations

    def backwardPass(self, activations, y):
        """

        Parameters:
        - activations: List of activations from forward pass.
        - y: True labels (one-hot encoded).

        Returns:
        - grad_w: Gradients for weights.
        - grad_b: Gradients for biases.
        """
        grad_w = [None] * (self.num_layers - 1)
        grad_b = [None] * (self.num_layers - 1)

        # Compute output error
        delta = activations[-1] - y
        grad_w[-1] = np.dot(activations[-2].T, delta)
        grad_b[-1] = np.sum(delta, axis=0, keepdims=True)

        # Backpropagate through hidden layers
        for i in range(self.num_layers - 3, -1, -1):
            delta = np.dot(delta, self.weights[i+1].T) * self.sigmoid_derivative(activations[i+1])
            grad_w[i] = np.dot(activations[i].T, delta)
            grad_b[i] = np.sum(delta, axis=0, keepdims=True)

        return grad_w, grad_b

    def updateWeights(self, grad_w, grad_b):
        """

        Parameters:
        - grad_w: Gradients for weights.
        - grad_b: Gradients for biases.
        """
        for i in range(len(self.weights)):
            self.velocities_w[i] = self.momentum * self.velocities_w[i] - self.learning_rate * grad_w[i]
            self.weights[i] += self.velocities_w[i]

            self.velocities_b[i] = self.momentum * self.velocities_b[i] - self.learning_rate * grad_b[i]
            self.biases[i] += self.velocities_b[i]

    def train(self, X, y, iterations=1000):
        """

        Parameters:
        - X: Training data.
        - y: Training labels (one-hot encoded).
        - iterations: Number of training iterations.
        """
        for i in range(iterations):
            activations = self.forwardPass(X)
            grad_w, grad_b = self.backwardPass(activations, y)
            self.updateWeights(grad_w, grad_b)

            if (i + 1) % 100 == 0:
                loss = -np.mean(np.sum(y * np.log(activations[-1] + 1e-8), axis=1))
                print(f"Iteration {i + 1}/{iterations}, Loss: {loss:.4f}")

    def predict(self, X):
        activations = self.forwardPass(X)
        return np.argmax(activations[-1], axis=1)
