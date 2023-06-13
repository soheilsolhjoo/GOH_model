import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

# Define the input data
X = np.random.rand(100, 3)  # Random input data with shape (100, 3)

# Define the target dataset
Y = np.random.rand(100, 1)  # Random target dataset with shape (100, 1)

# Create a custom loss function
def custom_loss(y_true, y_pred):
    y_true_col1 = y_true[:, 0]
    y_pred_col1 = y_pred[:, 0]
    derivative = tf.gradients(y_pred_col1, [X])[0]
    loss = tf.reduce_mean(tf.square(y_true_col1 - y_pred_col1)) + tf.reduce_mean(tf.square(derivative[:, 0] - y_true_col1))
    return tf.cast(loss, tf.float64)  # Cast the loss to float64

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(4)
])

# Compile the model
model.compile(optimizer='adam', loss=custom_loss)

# Define a function to return the loss and gradients
@tf.function
def get_loss_and_gradients(X, Y):
    with tf.GradientTape() as tape:
        predictions = model(X)
        loss = custom_loss(Y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    return tf.cast(loss, tf.float64), [tf.cast(grad, tf.float64) for grad in gradients]  # Cast loss and gradients to float64

# Define a function to pass to SciPy's minimize
def scipy_loss_and_gradients(x):
    X_batch = tf.convert_to_tensor(x.reshape(1, -1), dtype=tf.float32)
    Y_batch = tf.zeros((1, 4), dtype=tf.float32)  # Placeholder Y value
    loss, gradients = get_loss_and_gradients(X_batch, Y_batch)
    return loss.numpy(), np.concatenate([g.numpy().flatten() for g in gradients])

# Training loop
for epoch in range(10):
    loss_value, gradients_value = get_loss_and_gradients(X, np.hstack((Y, np.zeros((len(X), 3)))))
    optimizer_result = minimize(scipy_loss_and_gradients, x0=X.flatten(), jac=True, method='L-BFGS-B')
    model.set_weights(optimizer_result.x.reshape(model.get_weights()[0].shape))

# Optimal output calculation
optimal_output = model.predict(X)
