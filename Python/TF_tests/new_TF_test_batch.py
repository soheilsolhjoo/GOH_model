import tensorflow as tf
import numpy as np

# Generate random example data
x_train = np.random.rand(100, 10)
y_train = np.random.rand(100, 1)
c = np.random.rand(1000, 1)

# Generate indices corresponding to the training data
train_indices = np.arange(len(x_train))

# Define a custom loss function
def custom_loss(c):
    def loss_function(y_true, y_pred):
        mse_loss_1 = tf.reduce_mean(tf.square(y_true[:, 0] - y_pred[:, 0]))

        selected_indices = loss_inputs['indices']
        mse_loss_2 = tf.reduce_mean(tf.square(tf.cast(tf.gather(c, selected_indices), dtype=tf.float32) - y_pred[:, 1]))

        return mse_loss_1 + mse_loss_2

    return loss_function

# Define your network architecture
class CustomModel(tf.keras.Model):
    def __init__(self, input_neurons, output_neurons, num_hidden_layers, hidden_neurons):
        super(CustomModel, self).__init__()
        self.dense_input = tf.keras.layers.Dense(hidden_neurons, activation='relu', input_shape=(input_neurons,))
        self.hidden_layers = []
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(tf.keras.layers.Dense(hidden_neurons, activation='relu'))
        self.dense_output = tf.keras.layers.Dense(output_neurons)

    def call(self, inputs, training=False):
        x, indices = inputs
        x = self.dense_input(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        x = self.dense_output(x)
        return x



# Create a dictionary of loss inputs
loss_inputs = {
    'indices': train_indices
}

# Create an instance of the custom model
input_neurons = 10
output_neurons = 2
hidden_neurons = 64
num_hidden_layers = 3
model = CustomModel(input_neurons, output_neurons, num_hidden_layers, hidden_neurons)

# model = CustomModel()

# Compile the model with custom loss function
model.compile(optimizer='adam', loss=custom_loss(c))

# Train the model
model.fit([x_train, train_indices], y_train, epochs=20, batch_size=16)
