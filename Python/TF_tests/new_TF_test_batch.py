import tensorflow as tf
import numpy as np
import pandas as pd

# Generate random example data
x_train = np.random.rand(100, 10)
c_train = np.random.rand(100, 1)
y_train = np.random.rand(100, 1)

print(c_train)

x_in = np.concatenate([c_train, x_train], axis=1)

# Define the CustomModel class
class CustomModel(tf.keras.Model):
    def __init__(self, architecture):
        super(CustomModel, self).__init__()
        self.model = tf.keras.Sequential(architecture)

    def call(self, inputs):
        return self.model(inputs)

    def train_step(self, data):
        x, y = data

        # Extract the first column of x
        first_column = x[:, 0]
        x = x[:, 1:]

        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(x, training=True)

            # Compute the loss
            loss = custom_loss(y, y_pred, first_column)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dictionary with metrics
        return {m.name: m.result() for m in self.metrics}

# Define a custom loss function
def custom_loss(y_true, y_pred, first_column):
    # Compute the loss using y_true, y_pred, and first_column
    # You can define your custom loss computation here
    # For example, mean squared error multiplied by the sum of the first_column
    loss = tf.reduce_mean(tf.square(y_true - y_pred)) * tf.reduce_sum(first_column)
    return loss

# Define the desired architecture
architecture = [
    tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
]

# Create an instance of CustomModel
model = CustomModel(architecture)

# Compile the model
model.compile(optimizer='adam', loss=custom_loss)

# Train the model
model.fit(x_in, y_train, epochs=20, batch_size=16)
