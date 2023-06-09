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
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from custom_loss import custom_loss # type: ignore

from sklearn.preprocessing import MinMaxScaler
import joblib

# import numpy as np
# import matplotlib.pyplot as plt
# import autograd.numpy as np
# from autograd import jacobian
# # import scipy
# from scipy.optimize import minimize
# from scipy.optimize import Bounds

def main(data_dir):
    # Assign the file names and specify the file paths
    current_dir  = os.path.dirname(os.path.abspath(__file__))
    NN_file_path = os.path.join(current_dir, "GOH_NN.h5")
    
    folder_path = 'data'
    os.makedirs(folder_path, exist_ok=True)
    dataFile_eq  = os.path.join(current_dir, "data\\data_eq.csv")
    dataFile_x  = os.path.join(current_dir, "data\\data_x.csv")
    dataFile_y  = os.path.join(current_dir, "data\\data_y.csv")
    dataFile_g  = os.path.join(current_dir, "data\\g.csv")


    # Check if the file exists
    if os.path.exists(dataFile_eq):
        # Load the DataFrame from the saved files
        data_eq = pd.read_csv(dataFile_eq)
        data_x = pd.read_csv(dataFile_x)
        data_y = pd.read_csv(dataFile_y)
        g = np.loadtxt(dataFile_g, delimiter=',')
    else:
        # data_dir = "C:\\Users\P268670\Documents\Work\git\GOH_model\dataset\\"
        data_eq = data_dir+'Subject111_Sample1_YoungDorsal_Equibiaxial.csv'
        data_x = data_dir+'Subject111_Sample1_YoungDorsal_OffbiaxialX.csv'
        data_y = data_dir+'Subject111_Sample1_YoungDorsal_OffbiaxialY.csv'
        
        # The output of the data_preparation function has the folloing column_names:
        # ['Lambda11(-)', 'Lambda22(-)', 'Sigma11(KPa)', 'Sigma22(KPa)', 'Energy_exp', 'Lambda33(-)', 'I1', 'I41', 'I42']

        data_eq, g = f.data_preparation(data_eq)
        data_x = f.data_preparation(data_x)[0]
        data_y = f.data_preparation(data_y)[0]

        # Save the DataFrame to the file
        data_eq.to_csv(dataFile_eq, index=False)
        data_x.to_csv(dataFile_x, index=False)
        data_y.to_csv(dataFile_y, index=False)
        g = np.savetxt(dataFile_g, g, delimiter=',')
    
    # define column names for later data collection
    I_col = ["I1", "I41", "I42"]
    W_col = ["Energy_exp"]
    L_col = ['Lambda11(-)', 'Lambda22(-)', 'Lambda33(-)']
    S_col = ['Sigma11(KPa)', 'Sigma22(KPa)']

    if not os.path.exists(NN_file_path):
        # collect data
        X_train = pd.concat([data_x[I_col], data_y[I_col]])
        y_train = pd.concat([data_x[W_col], data_y[W_col]])
        X_eval  = data_eq[I_col]
        y_eval  = data_eq[W_col]
        lambda_train = pd.concat([data_x[L_col], data_y[L_col]]).values
        cauchy_train = pd.concat([data_x[S_col], data_y[S_col]]).values
        lambda_eval  = data_eq[L_col].values
        cauchy_eval  = data_eq[S_col].values

        # # Scale the input values
        # scaler = MinMaxScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_eval = scaler.transform(X_eval)
        # joblib.dump(scaler, 'GOH_NN.scaler')

        # Create TensorFlow variables for input data
        input_train = tf.Variable(X_train)
        input_eval = tf.Variable(X_eval)

        # Creating a NN model
        model = Sequential()
        model.add(Dense(3,activation='relu')) #inputs: I1, I41, I42
        model.add(Dense(8,activation='relu'))
        model.add(Dense(4,activation='linear')) #outputs: W, dWI1, dWI41, dWI42

        # Compile the model with custom loss function
        model.compile(optimizer='adam', loss=custom_loss(model, input_train, input_eval, cauchy_train, cauchy_eval, lambda_train, lambda_eval, g))

        # Training the model
        model.fit(x=X_train,y=y_train.values,
            validation_data=(X_eval,y_eval.values),
            batch_size=8, epochs=512)
        
        # Plot the losses
        losses = pd.DataFrame(model.history.history)
        losses.plot()
        plt.show()

        model.save('GOH_NN.h5')
    else:
        model = load_model('GOH_NN.h5', compile=False)
        # scaler = joblib.load('GOH_NN.scaler')

    # Evaluate the trained NN
    lambdas = data_eq[L_col].values
    invs    = data_eq[I_col].values
    # invs    = scaler.transform(invs)
    dWI     = model(invs)[:, 1:]
    stress  = f.WI_stress_NN_train(lambdas,invs,g,dWI)

    # Test data on graph
    sigmas  = data_eq[S_col].values
    plt.plot(lambdas[:, 0], sigmas[:, 0], 'wo', markeredgecolor='b', label='X - data')
    plt.plot(lambdas[:, 1], sigmas[:, 1], 'wo', markeredgecolor='r', label='Y - data')
    plt.plot(lambdas[:, 0], stress[:, 0], 'b-', label='X - NN')
    plt.plot(lambdas[:, 1], stress[:, 1], 'r-', label='Y - NN')
    plt.xlabel('stretch')
    plt.ylabel('stress (KPa)')
    # plt.title('X-Y Data')
    plt.show()


if __name__ == "__main__":
    data_dir = "C:\\Users\P268670\Documents\Work\git\GOH_model\dataset\\"
    main(data_dir)