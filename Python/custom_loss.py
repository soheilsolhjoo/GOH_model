import tensorflow as tf
import tensorflow.keras.backend as K
from consts import *
import functions as f

def custom_loss(model, input_train, input_eval, cauchy_train, cauchy_eval, lambda_train, lambda_eval, g):
    def loss_function(y_true, y_pred):
        ## Energy Loss
        mse_W = tf.reduce_mean(tf.square(y_true[:, 0] - y_pred[:, 0]))  # Mean squared error for energy

        ## data prep for derivative and cauchy stress losses
        if K.learning_phase() == 0:  # Training phase
            X = input_train
            Y = cauchy_train
            lambdas = lambda_train
        else:    # Evaluation phase
            X = input_eval
            Y = cauchy_eval
            lambdas = lambda_eval

        ## Derivative Losses
        out = model(X)
        dWI = tf.gradients(out[:, 0], X)[0]
        dWI = tf.cast(dWI, tf.float32)
        # mse_dWI1 = tf.reduce_mean(tf.square(dWI[:, 0] - out[:, 1]))  # Mean squared error with derivative
        # mse_dWI2 = tf.reduce_mean(tf.square(dWI[:, 1] - out[:, 2]))  # Mean squared error with derivative
        # mse_dWI3 = tf.reduce_mean(tf.square(dWI[:, 2] - out[:, 3]))  
        # mse_dWI = mse_dWI1 + mse_dWI2 + mse_dWI3
        # Or concicely:
        mse_dWI = tf.reduce_mean(tf.square(dWI - out[:, 1:]))  # Mean squared error for derivative

        L1 = mse_W + mse_dWI

        ## Stress loss
        stress  = f.WI_stress_NN_train(lambdas,X,g,dWI)
        # L_stress = tf.reduce_mean(tf.square(Y - stress))  # Mean squared error for stress
        L_stress = tf.sqrt(tf.reduce_sum(tf.square(Y - stress))) # Forbenius norm of Y - stress

        L2 = L_stress

        a1 = 0.1
        a2 = 10

        total_loss = a1*L1 + a2*L2
        return total_loss  # Combine the two losses

    return loss_function