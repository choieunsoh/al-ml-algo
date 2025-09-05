import torch
import torch.nn as nn
import torch.optim as optim


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNN()
optimizer = optim.SGD(model.parameters(), lr=0.01)

import tensorflow as tf

# Define a simple feedforward neural network in TensorFlow
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10)
])

# Compile the model
model.compile(optimizer='sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy())
