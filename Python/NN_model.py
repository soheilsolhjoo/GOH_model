# Import packages
import functions as f
from consts import *
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam

# import numpy as np
# import matplotlib.pyplot as plt
# import autograd.numpy as np
# from autograd import jacobian
# # import scipy
# from scipy.optimize import minimize
# from scipy.optimize import Bounds


def main(data_dir):
    # data_dir = "C:\\Users\P268670\Documents\Work\git\GOH_model\dataset\\"
    data_eq = data_dir+'Subject111_Sample1_YoungDorsal_Equibiaxial.csv'
    data_x = data_dir+'Subject111_Sample1_YoungDorsal_OffbiaxialX.csv'
    data_y = data_dir+'Subject111_Sample1_YoungDorsal_OffbiaxialY.csv'

    # The output of the data_preparation function has the folloing column_names:
    # ['Lambda11(-)', 'Lambda22(-)', 'Sigma11(KPa)', 'Sigma22(KPa)', 'Energy_exp', 'Lambda33(-)', 'I1', 'I41', 'I42']

    data_eq, g = f.data_preparation(data_eq)
    data_x = f.data_preparation(data_x)[0]
    data_y = f.data_preparation(data_y)[0]

    # Collecting Data
    I_col = ["I1", "I41", "I42"]
    W_col = ["Energy_exp"]
    X_train = pd.concat([data_x[I_col], data_y[I_col]], ignore_index=True)
    y_train = pd.concat([data_x[W_col], data_y[W_col]], ignore_index=True)
    X_eval  = data_eq[I_col]
    y_eval  = data_eq[W_col]
    
    # Creating a NN model
    model = Sequential()
    model.add(Dense(3,activation='relu')) #inputs: I1, I41, I42
    model.add(Dense(8,activation='relu'))
    model.add(Dense(1)) #outputs: W, dWI1, dWI41, dWI42
    model.compile(optimizer='adam',loss='mse')

    # Training the model
    model.fit(x=X_train.values,y=y_train.values,
          validation_data=(X_eval.values,y_eval.values),
          batch_size=128,epochs=400)
    
    # Plot the losses
    losses = pd.DataFrame(model.history.history)
    losses.plot()


if __name__ == "__main__":
    data_dir = "C:\\Users\P268670\Documents\Work\git\GOH_model\dataset\\"
    main(data_dir)