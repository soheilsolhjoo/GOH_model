# Import packages
import os
import functions as f
from consts import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

from sklearn.preprocessing import MinMaxScaler
import joblib

def main(data_dir):
    # Assign the file names and specify the file paths
    current_dir  = os.path.dirname(os.path.abspath(__file__))
    NN_file_path = os.path.join(current_dir, "GOH_NN.h5")
    
    data_eq, data_x, data_y, g = f.read_data(current_dir)
    
    if not os.path.exists(NN_file_path):
        # collect data
        X_train         = pd.concat([data_eq[I_col], data_x[I_col], data_y[I_col]]).values
        y_train         = pd.concat([data_eq[W_col], data_x[W_col], data_y[W_col]]).values
        lambda_train    = pd.concat([data_eq[L_col], data_x[L_col], data_y[L_col]]).values
        cauchy_train    = pd.concat([data_eq[S_col], data_x[S_col], data_y[S_col]]).values

        # # Scale the input values
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        # X_eval = scaler.transform(X_eval)
        joblib.dump(scaler, 'GOH_NN.scaler')

        # Create TensorFlow variables for input data
        inv_train = tf.Variable(X_train)
        # input_eval = tf.Variable(X_eval)

        # Creating a NN model
        model = Sequential()
        act_fun = 'tanh'
        model.add(Dense(3,activation=act_fun)) #inputs: I1, I41, I42
        n_layer = 8
        model.add(Dense(n_layer,activation=act_fun))
        model.add(Dense(n_layer,activation=act_fun))
        model.add(Dense(n_layer,activation=act_fun))
        model.add(Dense(1,activation='linear')) #outputs: W, dWI1, dWI41, dWI42

        # Compile the model with custom loss function
        model.compile(optimizer='adam',
                      loss=f.custom_loss(model, inv_train, cauchy_train, lambda_train, y_train, g),
                    #   metrics=[f.custom_metric]
                      )

        # Setup early stop
        # early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

        # Training the model
        model.fit(x=X_train,y=y_train,
            # validation_data=(X_eval,y_eval.values),
            # batch_size=8
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
    f.plot(data_x,  g,'NN_x.svg' ,'NN_offX'        ,'NN',model=model,scaler=scaler,stress_fig=False,energy_fig=True)
    f.plot(data_y,  g,'NN_y.svg' ,'NN_offX'        ,'NN',model=model,scaler=scaler,stress_fig=False,energy_fig=True)
    f.plot(data_eq, g,'NN_eq.svg','NN_equibiaxial' ,'NN',model=model,scaler=scaler,stress_fig=False,energy_fig=True)

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