# Import packages
import os
import functions as f
from consts import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

from sklearn.preprocessing import MinMaxScaler
import joblib

def main(data_dir):
    # Assign the file names and specify the file paths
    current_dir  = os.path.dirname(os.path.abspath(__file__))
    NN_file_path = os.path.join(current_dir, "GOH_NN.h5")
    
    data_eq, data_x, data_y, g = f.read_data(current_dir,data_dir)
    G41 = tf.cast(tf.einsum('i,j->ij', g[0,:], g[0,:]), tf.float32)
    G42 = tf.cast(tf.einsum('i,j->ij', g[1,:], g[1,:]), tf.float32)
    
    if not os.path.exists(NN_file_path):
        ## collect data
        X_train         = pd.concat([data_eq[I_col], data_x[I_col], data_y[I_col]]).values
        y_train         = pd.concat([data_eq[W_col], data_x[W_col], data_y[W_col]]).values
        lambda_train    = pd.concat([data_eq[L_col], data_x[L_col], data_y[L_col]]).values
        cauchy_train    = pd.concat([data_eq[S_col], data_x[S_col], data_y[S_col]]).values

        # X_train         = pd.concat([data_eq[I_col]]).values
        # y_train         = pd.concat([data_eq[W_col]]).values
        # lambda_train    = pd.concat([data_eq[L_col]]).values
        # cauchy_train    = pd.concat([data_eq[S_col]]).values

        # # Shuffle data
        # np.random.shuffle(X_train)
        # np.random.shuffle(y_train)
        # np.random.shuffle(lambda_train)
        # np.random.shuffle(cauchy_train)


        # # Scale the input values
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        # # X_eval = scaler.transform(X_eval)
        joblib.dump(scaler, 'GOH_NN.scaler')

        # Create TensorFlow variables for input data
        # inv_train = tf.Variable(X_train)
        # input_eval = tf.Variable(X_eval)

        # Creating a NN model
        model = Sequential()
        act_fun = 'relu'
        model.add(Dense(3,activation=act_fun)) #inputs: I1, I41, I42
        n_neurons = 8
        for i in range(3):
            model.add(Dense(n_neurons,activation=act_fun))
            model.add(Dropout(0.2))
        model.add(Dense(4,activation='linear')) #outputs: W, dWI1, dWI41, dWI42

        # Compile the model with custom loss function
        # learning_rate = 0.0001  # Specify your desired learning rate
        # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
                      optimizer='adam',
                    #   optimizer=optimizer,
                      loss=f.custom_loss(model, tf.Variable(tf.convert_to_tensor(X_train)), y_train, cauchy_train, lambda_train,  G41,G42),
                    #   metrics=[f.custom_metric]
                      )

        # Setup early stop
        # early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

        # Training the model
        model.fit(x=X_train,y=y_train,
            # validation_data=(X_eval,y_eval.values),
            batch_size=32
            epochs=1000,
            # callbacks=[early_stop]
            )
        
        # Plot the losses
        losses = pd.DataFrame(model.history.history)
        losses.plot()
        plt.yscale('log')
        plt.show()

        model.save('GOH_NN.h5')
    else:
        model = load_model('GOH_NN.h5', compile=False)
        scaler = joblib.load('GOH_NN.scaler')

    # Evaluate the trained NN
    # f.plot(data_x,  g,'NN_x.svg',   'NN_offX',        method='NN',model=model,scaler=scaler)
    # f.plot(data_y,  g,'NN_y.svg',   'NN_offY',        method='NN',model=model,scaler=scaler)
    # f.plot(data_eq, g,'NN_eq.svg',  'NN_equibiaxial', method='NN',model=model,scaler=scaler)

    f.plot(data_x,  g,'NN_x.svg' ,'NN_offX'        ,method='NN',model=model,scaler=scaler,stress_fig=True,energy_fig=True)
    f.plot(data_y,  g,'NN_y.svg' ,'NN_offX'        ,method='NN',model=model,scaler=scaler,stress_fig=True,energy_fig=True)
    f.plot(data_eq, g,'NN_eq.svg','NN_equibiaxial' ,method='NN',model=model,scaler=scaler,stress_fig=True,energy_fig=True)

    lambda_space = f.lambda_space_generator(g, max_lambda=1.3, number_test_points=50)
    f.plot(lambda_space,  g, method='NN',model=model,scaler=scaler,lambda_space=True)

    # lambdas = data_eq[L_col].values
    # invs    = data_eq[I_col].values
    # invs    = scaler.transform(invs)
    # dWI     = model(invs)[:, 1:]
    # stress  = f.WI_stress_NN_train(lambdas,invs,g,dWI)

    # # Test data on graph
    # sigmas  = data_eq[S_col].values
    # plt.plot(lambdas[:, 0], sigmas[:, 0], 'wo', markeredgecolor='b', label='X - data')
    # plt.plot(lambdas[:, 1], sigmas[:, 1], 'wo', markeredgecolor='r', label='Y - data')
    # plt.plot(lambdas[:, 0], stress[:, 0], 'b-', label='X - NN')
    # plt.plot(lambdas[:, 1], stress[:, 1], 'r-', label='Y - NN')
    # plt.xlabel('stretch')
    # plt.ylabel('stress (KPa)')
    # # plt.title('X-Y Data')
    # plt.show()


if __name__ == "__main__":
    data_dir = "C:\\Users\P268670\Documents\Work\git\GOH_model\dataset\\"
    main(data_dir)