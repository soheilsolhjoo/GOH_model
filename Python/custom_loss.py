import tensorflow as tf
import tensorflow.keras.backend as K

def custom_loss(model, input_train, input_eval):
    def loss_function(y_true, y_pred):
        ## Energy Loss
        mse_W = tf.reduce_mean(tf.square(y_true[:, 0] - y_pred[:, 0]))  # Mean squared error

        ## Derivative Losses
        if K.learning_phase() == 0:  # Training phase
            X = input_train
        else:    # Evaluation phase
            X = input_eval

        out = model(X)
        dWI = tf.gradients(out[:, 0], X)[0]
        dWI = tf.cast(dWI, tf.float32)
        # mse_dWI1 = tf.reduce_mean(tf.square(dWI[:, 0] - out[:, 1]))  # Mean squared error with derivative
        # mse_dWI2 = tf.reduce_mean(tf.square(dWI[:, 1] - out[:, 2]))  # Mean squared error with derivative
        # mse_dWI3 = tf.reduce_mean(tf.square(dWI[:, 2] - out[:, 3]))  
        # mse_dWI = mse_dWI1 + mse_dWI2 + mse_dWI3
        # Or concicely:
        mse_dWI = tf.reduce_mean(tf.square(dWI - out[:, 1:]))  # Mean squared error with derivative

        return mse_W + mse_dWI  # Combine the two losses

    return loss_function