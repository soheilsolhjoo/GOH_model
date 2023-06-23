import tensorflow as tf
import numpy as np

# Generate random example data
x_data = np.random.rand(100, 10)
y_data = np.random.rand(100, 1)
c_data = np.random.rand(1000, 1)

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

# Number of folds for cross-validation
num_folds = 5

# Split the data into train/validation folds
fold_size = len(x_data) // num_folds
fold_indices = np.arange(len(x_data))
np.random.shuffle(fold_indices)

# Perform cross-validation
for fold in range(num_folds):
    print(f"Fold {fold+1}/{num_folds}")

    # Split the data into train/validation sets
    validation_indices = fold_indices[fold * fold_size: (fold + 1) * fold_size]
    train_indices = np.concatenate([fold_indices[:fold * fold_size], fold_indices[(fold + 1) * fold_size:]])

    # Create a dictionary of loss inputs
    loss_inputs = {'indices': train_indices}

    # Create an instance of the custom model
    input_neurons = 10
    output_neurons = 2
    hidden_neurons = 64
    num_hidden_layers = 3
    model = CustomModel(input_neurons, output_neurons, num_hidden_layers, hidden_neurons)

    # Compile the model with custom loss function
    model.compile(optimizer='adam', loss=custom_loss(c_data))

    # Train the model
    x_train_fold = x_data[train_indices]
    y_train_fold = y_data[train_indices]
    model.fit([x_train_fold, train_indices], y_train_fold, epochs=20, batch_size=16)

    # Evaluate the model on the validation set
    x_val_fold = x_data[validation_indices]
    y_val_fold = y_data[validation_indices]
    loss = model.evaluate([x_val_fold, validation_indices], y_val_fold)
    print("Validation loss:", loss)
    print()
