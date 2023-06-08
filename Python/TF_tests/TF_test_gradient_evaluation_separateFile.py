import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from custom_loss_test import custom_loss
import os
from tensorflow.keras.models import load_model

# Assign name for saving the network
current_dir = os.path.dirname(os.path.abspath(__file__))
NN_file_path = os.path.join(current_dir, "trained_NN.h5")

# Generate random example data
x_train = np.random.rand(100, 10)
y_train = np.random.rand(100, 1)  # Modify output dimension to 2

x_eval = np.random.rand(50, 10)
y_eval = np.random.rand(50, 1)

# Create TensorFlow variables for input data
input_train = tf.Variable(x_train)
input_eval = tf.Variable(x_eval)


def load_model_with_custom_loss(loss_function, input_train, input_eval):
    return load_model(NN_file_path, custom_objects={'custom_loss': loss_function})



if not os.path.exists(NN_file_path):
    # Define your network architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2)  # Modify output dimension to 2
    ])

    # Compile the model with custom loss function
    model.compile(optimizer='adam', loss=custom_loss(model, input_train, input_eval))

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=32, 
            validation_data=(x_eval, y_eval))

    # Plot the model's fitting history
    losses = pd.DataFrame(model.history.history)
    losses.plot()
    plt.show()

    model.save(NN_file_path)

else:
    # model = load_model(NN_file_path, custom_objects={'custom_loss': custom_loss(model, input_train, input_eval)})
    # model = load_model_with_custom_loss(custom_loss, input_train, input_eval)
    model = load_model(NN_file_path, compile=False)


y_pred = model.predict(x_eval)
plt.plot(y_pred[:, 0], y_eval, 'bo')
plt.show()