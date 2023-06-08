##############################################
# 1D - 1 loss
##############################################
# import tensorflow as tf
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # Define a custom loss function
# def custom_loss(y_true, y_pred):
#     return tf.reduce_mean(tf.square(y_true - y_pred))  # Mean squared error

# # Generate random example data
# x_train = np.random.rand(100, 10)
# y_train = np.random.rand(100, 1)

# # Define your network architecture
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1)
# ])

# # Compile the model with custom loss function
# model.compile(optimizer='adam', loss=custom_loss)

# # Train the model
# model.fit(x_train, y_train, epochs=10, batch_size=32)

# losses = pd.DataFrame(model.history.history)
# losses.plot()
# plt.show()
##############################################
# 2D - 1 loss
##############################################
# import tensorflow as tf
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # Define a custom loss function
# def custom_loss(y_true, y_pred):
#     return tf.reduce_mean(tf.square(y_true - y_pred))  # Mean squared error

# # Generate random example data
# x_train = np.random.rand(100, 10)
# y_train = np.random.rand(100, 2)  # Modify output dimension to 2

# # Define your network architecture
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(2)  # Modify output dimension to 2
# ])

# # Compile the model with custom loss function
# model.compile(optimizer='adam', loss=custom_loss)

# # Train the model
# model.fit(x_train, y_train, epochs=10, batch_size=32)

# losses = pd.DataFrame(model.history.history)
# losses.plot()
# plt.show()
##############################################
# 2D - 2 losses
##############################################
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define a custom loss function
def custom_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true[:, 0] - y_pred[:, 0]))  # Mean squared error
    mae_loss = tf.reduce_mean(tf.abs(y_true[:, 1] - y_pred[:, 1]))  # Mean absolute error
    return mse_loss + mae_loss  # Combine the two losses

# Generate random example data
x_train = np.random.rand(100, 10)
y_train = np.random.rand(100, 2)  # Modify output dimension to 2

# Define your network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2)  # Modify output dimension to 2
])

# Compile the model with custom loss function
model.compile(optimizer='adam', loss=custom_loss)

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)

losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()
##############################################
# 2D - physical problem
##############################################
# import tensorflow as tf
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # Define a custom loss function with physics insights
# def custom_loss(y_true, y_pred):
#     mse_loss = tf.reduce_mean(tf.square(y_true[:, 0:2] - y_pred[:, 0:2]))  # Mean squared error for position and velocity
#     mae_loss = tf.reduce_mean(tf.abs(y_true[:, 0:2] - y_pred[:, 0:2]))  # Mean absolute error for position and velocity
    
#     # Additional physics insight: Penalize deviations from conservation of energy
#     energy_loss = tf.reduce_mean(tf.square(y_pred[:, 1] - (0.5 * y_pred[:, 0]**2)))  # Square of the energy conservation equation
    
#     return mse_loss + mae_loss + 0.1 * energy_loss  # Combine losses with an energy penalty term

# # Generate random example data for a particle's motion
# num_samples = 100
# time = np.linspace(0, 10, num_samples)
# position = 0.5 * time**2  # True position of the particle
# velocity = time  # True velocity of the particle

# # Combine position and velocity into the true label
# y_train = np.column_stack((position, velocity))

# # Generate random example data for the input features
# x_train = np.random.rand(num_samples, 10)

# # Define your network architecture
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(2)  # Output layer for position and velocity
# ])

# # Compile the model with custom loss function
# model.compile(optimizer='adam', loss=custom_loss)

# # Train the model
# model.fit(x_train, y_train, epochs=10, batch_size=32)

# # Generate predictions for the trained model
# predictions = model.predict(x_train)

# # Plot the true position and predicted position
# plt.plot(time, position, label='True Position')
# plt.plot(time, predictions[:, 0], label='Predicted Position')
# plt.xlabel('Time')
# plt.ylabel('Position')
# plt.legend()
# plt.show()

