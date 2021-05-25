from layer import Layer

from typing import List, Callable

class Network:
    def __init__(self):
        self.layers: List[Layer] = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer: Layer):
        """
        Add layer to network
        """
        self.layers.append(layer)

    def use(self, loss: Callable[[float], float], loss_prime):
        """
        Set loss to use
        """
        self.loss = loss
        self.loss_prime = loss_prime

    # predicto output fot given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculates average error in all samples
            err /= samples
            print(f"epoch {i + 1}/{epochs}  error = {err}")
