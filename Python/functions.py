# This code train a neural network to perform according to the GOH model.
# The code takes the following steps:
# - Prepare experimentel (stress-stretch) data
#     - load data
#     - rescale data
#     - calibrate the origin of data
# - Construct F (deformation gradient tensor)
# - Calculate C = F.F^T (!)
# - Calculate I1_exp and I4_exp
# - Calculate Phi (+ Phi_1 and Phi_4) for I1_exp and I4_exp
# - Assign collocation points for a range of stretches
#     - Calculate their corresponing I1_col and I4_col
# - Define Loss function for training "net"
#     - Calculate Phi, Phi_1, Phi_4 from net(I1_exp, I4_exp)
#     - Calculate GOH stress using the results in the previous step
#     - L1: f(phi and phi_derivatives)
#     - L2: f(stress)
#     - L3: f(phi_11, phi_14, phi_41, phi_44) : phi_14 = phi_41
#     - L = f(L1, L2, L3)
# - Train the network

# Import  packages
from consts import *
import re
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import autograd.numpy as np
from autograd import jacobian
# import scipy
from sklearn.preprocessing import MinMaxScaler
from scipy.integrate import cumtrapz
from scipy.optimize import minimize, Bounds
import tensorflow as tf
# from tensorflow import Tensor
import tensorflow.keras.backend as K
import keras
from consts import *
from varname import nameof


def MPa2KPa(data):
    """ convert MPa data to KPa """
    data['Sigma11(MPa)'] *= 1000
    data['Sigma22(MPa)'] *= 1000
    data = data.rename(columns={'Sigma11(MPa)': 'Sigma11(KPa)', 'Sigma22(MPa)': 'Sigma22(KPa)'})
    return data

def move2origin(data):
    """ move the origin to stretch = 1
    """
    data.drop(index=data.index[0], axis=0, inplace=True)
    # data['Lambda11(-)'] -= data['Lambda11(-)'][1] - 1
    # data['Lambda22(-)'] -= data['Lambda22(-)'][1] - 1
    # data['Sigma11(MPa)'] -= data['Sigma11(MPa)'][1]
    # data['Sigma22(MPa)'] -= data['Sigma22(MPa)'][1]
    data['Lambda11(-)'] -= min(data['Lambda11(-)']) - 1
    data['Lambda22(-)'] -= min(data['Lambda22(-)']) - 1
    data['Sigma11(MPa)'] -= min(data['Sigma11(MPa)'])
    data['Sigma22(MPa)'] -= min(data['Sigma22(MPa)'])
    return data

def C_W_exp(data):
    """ calculate the deformation energy
    """
    # W11 = data['Lambda11(-)'] * data['Sigma11(KPa)']
    # W22 = data['Lambda22(-)'] * data['Sigma22(KPa)']
    # W = W11 + W22
    # data['Energy_exp'] = W.cumsum()

    eps_11 = np.log(data['Lambda11(-)'])
    eps_22 = np.log(data['Lambda22(-)'])
    # W11 = np.trapz(data['Sigma11(KPa)'], x=eps_11)
    # W22 = np.trapz(data['Sigma22(KPa)'], x=eps_22)
    W11 = cumtrapz(data['Sigma11(KPa)'], x=eps_11, initial=0)
    W22 = cumtrapz(data['Sigma22(KPa)'], x=eps_22, initial=0)
    W = W11 + W22
    data['Energy_exp'] = W

    return data

def C_L33(data):
    """ calculate lambda33 = 1 / (lambda11 * lambda22)
    """
    data['Lambda33(-)'] = 1 / (data['Lambda11(-)'] * data['Lambda22(-)'])
    return data

def C_I1(data):
    """ calculate first invariant of the right cauchy-green tensor:
    I1 = lam11 ^ 2 + lam22 ^ 2 + lam33 ^ 2
    """
    data['I1'] =  np.power(data['Lambda11(-)'], 2) \
                + np.power(data['Lambda22(-)'], 2) \
                + np.power(data['Lambda33(-)'], 2)
    return data

def C_I4(data,g):
    """ calculate the pseudo-invariants I4_1 and I4_2
    I4 = g^2 * C
    """
    g = g[:,0:2] ** 2
    C = np.power(np.array([data['Lambda11(-)'],data['Lambda22(-)']]),2)
    I4s = (g @ C).T
    data['I41'] = I4s[:,0]
    data['I42'] = I4s[:,1]
    return data

def GOH_energy(const,I):
    """Calculates GOH energy, with:
    c : [mu, k1, k2, kappa]
    I : [I1, I4_1, I4_2]
    """
    mu, k1, k2, kappa = const
    W_iso = mu * (I[:,0] - 3)
    E = kappa * (I[:,0].reshape(-1, 1) - 3) + (1 - 3*kappa) * (I[:,1:] - 1)
    E = (abs(E) + E) / 2
    W_aniso = k1 / (2 * k2) * np.sum(np.exp((k2 * E**2) - 1), axis=1)
    return W_iso + W_aniso

def WI_stress_GOH(data,g,consts,del_I):
    """ calculate stress using the energy method based on invariants of tensor C
    """
    # collect lambdas and square them
    lam1_2 = data['Lambda11(-)'] ** 2
    lam2_2 = data['Lambda22(-)'] ** 2
    lam3_2 = data['Lambda33(-)'] ** 2
    # calculate derivative of W wrt I
    dWI_dic = {0:'dWI_1', 1:'dWI_41', 2:'dWI_42'}
    inv_list = ['I1','I41','I42']
    invs = data[inv_list]
    dWI = pd.DataFrame()
    c = consts
    for i in range(3):
        Is = [invs.copy() for _ in range(2)]
        Is[0][inv_list[i]] += del_I
        Is[1][inv_list[i]] -= del_I
        dWI[dWI_dic[i]] = (GOH_energy(c,Is[0].values) - GOH_energy(c,Is[1].values)) / (2*del_I)
    # loop over data points
    sigma = np.empty((data.shape[0],2))
    for i in range(data.shape[0]):
        # calculate S
        # print(i)
        S1 = dWI[dWI_dic[0]][i] * np.eye(3)
        S2 = dWI[dWI_dic[1]][i] * np.outer(g[0,:],g[0,:])
        S3 = dWI[dWI_dic[2]][i] * np.outer(g[1,:],g[1,:])
        S_PK2 = 2 * (S1 + S2 + S3)

        # find pressure
        # calculate Cauchy stresses
        # p = lam3_2[i+1] * S_PK2[2,2]
        # sigma[i,0] = lam1_2[i+1] * S_PK2[0,0] - p
        # sigma[i,1] = lam2_2[i+1] * S_PK2[1,1] - p
        # OR: do them in one line
        sigma[i,:] = [lam1_2[i] * S_PK2[0,0] , lam2_2[i] * S_PK2[1,1]] - (lam3_2[i] * S_PK2[2,2])
    
    # sigma = sigma.round(decimals=3)
    return sigma
    
def read_data(current_dir,data_dir):
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

        data_eq, g = data_preparation(data_eq)
        data_x = data_preparation(data_x)[0]
        data_y = data_preparation(data_y)[0]

        # Save the DataFrame to the file
        data_eq.to_csv(dataFile_eq, index=False)
        data_x.to_csv(dataFile_x, index=False)
        data_y.to_csv(dataFile_y, index=False)
        g = np.savetxt(dataFile_g, g, delimiter=',')

    return data_eq, data_x, data_y, g

def WI_stress_NN_train(lambdas,invs,dWI,G41,G42):
    """ calculate stress using the energy method based on invariants of tensor C
    """
    # collect lambdas and square them
    lambdas_2 = lambdas ** 2
    # cast variables into tf.float32
    lambdas_2 = tf.cast(lambdas_2, tf.float32)
    # g = tf.cast(g, tf.float32)
    # pre-allocate memory for sigma
    # sigma = np.empty((invs.shape[0],2))
    # sigma = tf.Variable(tf.zeros((invs.shape[0], 2), dtype=tf.float32)) # Raise: ValueError: tf.function only supports singleton tf.Variables created on the first call. Make sure the tf.Variable is only created once or created outside tf.function.
    sigma = tf.TensorArray(dtype=tf.float32, size=invs.shape[0], dynamic_size=False)
    # loop over data points
    # G41 = tf.einsum('i,j->ij', g[0,:], g[0,:])
    # G42 = tf.einsum('i,j->ij', g[1,:], g[1,:])

    for i in range(invs.shape[0]):
        # calculate S
        S1 = dWI[i,0] * tf.eye(3)
        S2 = dWI[i,1] * G41 #np.outer(g[0,:],g[0,:])
        S3 = dWI[i,2] * G42 #np.outer(g[1,:],g[1,:])
        S_PK2 = 2 * (S1 + S2 + S3)

        # calculate pressure
        p = lambdas_2[i,2] * S_PK2[2,2]
        # calculate Cauchy stresses
        sigma = sigma.write(i, [
            lambdas_2[i, 0] * S_PK2[0, 0] - p,
            lambdas_2[i, 1] * S_PK2[1, 1] - p
        ])
    sigma = sigma.stack()

    return sigma



def WI_stress_NN(lambdas,invs,g,dWI):
    """ calculate stress using the energy method based on invariants of tensor C
    """
    # collect lambdas and square them
    lambdas_2 = lambdas ** 2
    # lam2_2 = lambdas[:,1] ** 2
    # lam3_2 = lambdas[:,2] ** 2
    # # calculate derivative of W wrt I
    # dWI_dic = {0:'dWI_1', 1:'dWI_41', 2:'dWI_42'}
    # inv_list = ['I1','I41','I42']
    # invs = data[inv_list]
    # dWI = pd.DataFrame()
    # for i in range(3):
    #     Is = [invs.copy() for _ in range(2)]
    #     Is[0][inv_list[i]] += del_I
    #     Is[1][inv_list[i]] -= del_I
    #     dWI[dWI_dic[i]] = (GOH_energy(const,Is[0].values) - GOH_energy(const,Is[1].values)) / (2*del_I)
    # # loop over data points
    sigma = np.empty((invs.shape[0],2))
    for i in range(invs.shape[0]):
        # calculate S
        S1 = dWI[i,0] * np.eye(3)
        S2 = dWI[i,1] * np.outer(g[0,:],g[0,:])
        S3 = dWI[i,2] * np.outer(g[1,:],g[1,:])
        S_PK2 = 2 * (S1 + S2 + S3)
        # print("*: ", dWI[i,0])
        # print(S_PK2.consumers())
        # tf.print(S_PK2.eval())
        # tf.print(S_PK2, output_stream=sys.stderr)
        exit()
        # calculate Cauchy stresses
        sigma[i,:] = [lambdas_2[i,0] * S_PK2[0,0] , lambdas_2[i,1] * S_PK2[1,1]] - (lambdas_2[i,2] * S_PK2[2,2])
    
    # sigma = sigma.round(decimals=3)
    return sigma


def data_preparation(data_file):
    global alpha
    # ################################
    # # Pre-assigned data: to be modified by user
    # ################################
    # del_I   = 1e-6; # delta_I is used for calculating derivative of W (energy) wrt I (invariants), dWI
    # alpha = [0,90]  # fiber directions
    # # constants of the Gasser-Ogden-Holzapfel (GOH) model
    # const = [0.1,1,1,1/6] # [mu, k1, k2, kappa]

    ################################
    # The main code
    ################################
    # LOAD DATA
    data_ = pd.read_csv(data_file)
    # REPOSITION THE ORIGIN
    # Take this step with care, and ensure it's necessary for a physically meaningful behavior
    data_ = move2origin(data_)
    # RESCALE DATA
    data_ = MPa2KPa(data_)
    # Calculate elastic deformation energy
    data_ = C_W_exp(data_)
    # CONSTRUCT F : not needed. Instead only calculate lambda(33), L3 = 1/(L1*L2)
    data_ = C_L33(data_)
    # Calculate I1: I1 = L1^2 + L2^2 + L3^2
    data_ = C_I1(data_)
    
    # ASSIGN directions at alpha = 0 and 90 degrees
    alpha = np.array(alpha) * np.pi/180
    g = np.array([[np.cos(alpha[0]), np.sin(alpha[0]), 0],\
                  [np.cos(alpha[1]), np.sin(alpha[1]), 0]]).round(decimals=3)
    # Calculate psuedo-invariants for the directions g
    # I4i = g.^2 * C that is [2 x 2]*[2 x n_data]
    data_ = C_I4(data_,g)

    return data_, g

def tf_nan_check(variable,name):
    # if tf.math.is_nan(variable):
    if tf.math.reduce_any(tf.math.is_nan(variable)):
        # print(nameof(variable), variable)
        print(name,' : ', variable)

    
def custom_loss(model, inv_train, W_train, cauchy_train, lambda_train, G41, G42):
    def loss_function(y_true, y_pred):
        X = inv_train
        out = model(X)
        
        # ## Energy Loss
        W = tf.cast(W_train, dtype=out.dtype)
        mse_W = tf.reduce_mean(tf.square(W[:, 0] - out[:, 0]))  # Mean squared error for energy
        L1 = mse_W
        # Postivie energy
        abs_W = tf.reduce_mean(tf.square(tf.maximum(tf.math.negative(out[:, 0]), 0)))
        L1 += abs_W
        # ## Derivative Losses
        dWI = tf.cast(tf.gradients(out[:, 0], X)[0], tf.float32)
        mse_dWI = tf.reduce_mean(tf.square(dWI - out[:, 1:]))  # Mean squared error for derivative
        L1 += mse_dWI

        # ## Stress loss
        stress  = WI_stress_NN_train(lambda_train,X,out[:, 1:],G41,G42)        
        S = tf.cast(cauchy_train, dtype=stress.dtype)
        L_stress = tf.reduce_mean(tf.reduce_mean(tf.norm(S - stress, axis=1), axis=0))
        L2 = L_stress

        # ## convexity loss
        ddWI1, ddWI41, ddWI42 = [tf.cast(tf.gradients(out[:, i+1], X)[0], tf.float32) for i in range(3)]

        Hess = tf.transpose(tf.stack([ddWI1, ddWI41, ddWI42]), perm=[1, 0, 2])
        Hess_t= tf.transpose(Hess, perm=[0, 2, 1])

        # Hessain symmetry loss
        L_Hess = tf.reduce_mean(tf.reduce_sum(Hess - Hess_t, axis=[-2,-1]))
        L3 = L_Hess

        # # Calculate the minors
        # first_column = tf.gather(Hess[:, 0, 0], tf.range(Hess.shape[0]))
        # second_column = tf.linalg.det(Hess[:, :2, :2])
        # third_column = tf.linalg.det(Hess)
        # Delta_k = tf.stack([first_column, second_column, third_column], axis=1)
        # # convexity loss
        # L_positive = tf.reduce_mean(tf.reduce_sum(tf.maximum(tf.math.negative(Delta_k), 0), axis=1))
        # L3 += L_positive
        
        # a1 = 0.1
        # a2 = 1

        total_loss = L1 + L2 + L3
        return total_loss

    return loss_function

def custom_metric(y_true, y_pred):
    # Perform necessary calculations using TensorFlow operations
    metric_value = tf.reduce_mean(tf.square(y_true[:, 0] - y_pred[:, 0]))

    return metric_value


def const_read(const_file):
    # Read the text file
    with open(const_file, 'r') as file:
        lines = file.readlines()
    # Extract the values of x from the appropriate line
    for line in lines:
        match = re.search(r'\s*x:', line)
        if match:
            x_values_str = line.split(':')[1].strip()
            x_values_str = x_values_str.replace('[', '').replace(']', '')  # Remove square brackets
            X = [float(value) for value in x_values_str.split()]
            return X
    return None


def plot(data_,g,fig_name,title,method,consts=[],model=[],scaler=[],stress_fig=True,energy_fig=False):
    lambdas = data_[L_col].values
    
    if method == 'optimizer':
        stress = WI_stress_GOH(data_, g,consts,del_I)
    elif method == 'NN':
        invs    = data_[I_col].values
        invs    = scaler.transform(invs)

        # stress  = model(invs)
        NN_out = model(invs)
        W   = NN_out[:,0]
        dWI = NN_out[:, 1:]
        G41 = tf.cast(tf.einsum('i,j->ij', g[0,:], g[0,:]), tf.float32)
        G42 = tf.cast(tf.einsum('i,j->ij', g[1,:], g[1,:]), tf.float32)
        stress  = WI_stress_NN_train(lambdas,invs,dWI,G41,G42)
    
    if stress_fig:
        sigmas  = data_[S_col].values

        plt.plot(lambdas[:, 0], sigmas[:, 0], 'wo', markeredgecolor='b', label='X - data')
        plt.plot(lambdas[:, 1], sigmas[:, 1], 'wo', markeredgecolor='r', label='Y - data')
        plt.plot(lambdas[:, 0], stress[:, 0], 'b-', label='X - model')
        plt.plot(lambdas[:, 1], stress[:, 1], 'r-', label='Y - model')
        plt.legend()
        plt.xlabel('stretch')
        plt.ylabel('stress (KPa)')
        plt.title(title)
        plt.savefig(fig_name+'_S.svg', format='svg')
        plt.show()
    
    if energy_fig:
        energy  = data_[W_col].values

        plt.plot(energy, W, 'wo', markeredgecolor='r')
        plt.plot(energy, energy, 'b-')
        plt.xlabel('data')
        plt.ylabel('model')
        plt.title(title)
        plt.savefig(fig_name+'_E.svg', format='svg')
        plt.show()



if __name__ == "__main__":
    # data_dir = "C:\\Users\\P268670\\OneDrive - University of Groningen\\Documents\\Work\\git\\GOH_model\\"
    data_dir = "C:\\Users\P268670\Documents\Work\git\GOH_model"
    data_dir = data_dir + "\\dataset\\"
    data_file = data_dir+'Subject111_Sample1_YoungDorsal_Equibiaxial.csv'
    
    pass