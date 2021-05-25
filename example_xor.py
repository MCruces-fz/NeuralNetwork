import numpy as np

from network import Network
from fully_connected_layer import FCLayer
from activation_layer import ActivationLayer
from activation import tanh, tanh_prime
from loss import mse, mse_prime

# Training data
x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# Network
net = Network()
net.add(FCLayer(2, 3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(tanh, tanh_prime))

# Train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# Test
out = net.predict(x_train)
print(out)
