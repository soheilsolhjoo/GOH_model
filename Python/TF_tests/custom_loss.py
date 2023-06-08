import tensorflow as tf
import tensorflow.keras.backend as K

def custom_loss(model, input_train, input_eval):
    def loss_function(y_true, y_pred):
        mse_loss = tf.reduce_mean(tf.square(y_true[:, 0] - y_pred[:, 0]))  # Mean squared error

        if K.learning_phase() == 0:  # Training phase
            X = input_train
        else:    # Evaluation phase
            X = input_eval

        out = model(X)
        grad1 = tf.gradients(out[:, 0], X)[0][:, 1]
        grad1 = tf.cast(grad1, tf.float32)
        mae_loss = tf.reduce_mean(tf.abs(grad1 - out[:, 1]))  # Mean absolute error with derivative

        return mse_loss + mae_loss  # Combine the two losses

    return loss_function
