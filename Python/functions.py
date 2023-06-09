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
import os
import sys
import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import jacobian
# import scipy
from scipy.optimize import minimize
from scipy.optimize import Bounds
import tensorflow as tf
import tensorflow.keras.backend as K
from consts import *


def MPa2KPa(data):
    """ convert MPa data to KPa """
    data['Sigma11(MPa)'] *= 1
    data['Sigma22(MPa)'] *= 1
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
    W11 = data['Lambda11(-)'] * data['Sigma11(KPa)']
    W22 = data['Lambda22(-)'] * data['Sigma22(KPa)']
    W = W11 + W22
    data['Energy_exp'] = W.cumsum()
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

def WI_stress_GOH(data,g,const,del_I):
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
    for i in range(3):
        Is = [invs.copy() for _ in range(2)]
        Is[0][inv_list[i]] += del_I
        Is[1][inv_list[i]] -= del_I
        dWI[dWI_dic[i]] = (GOH_energy(const,Is[0].values) - GOH_energy(const,Is[1].values)) / (2*del_I)
    # loop over data points
    sigma = np.empty((data.shape[0],2))
    for i in range(data.shape[0]):
        # calculate S
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
        sigma[i,:] = [lam1_2[i+1] * S_PK2[0,0] , lam2_2[i+1] * S_PK2[1,1]] - (lam3_2[i+1] * S_PK2[2,2])
    
    # sigma = sigma.round(decimals=3)
    return sigma
    


def WI_stress_NN_train(lambdas,invs,g,dWI):
    """ calculate stress using the energy method based on invariants of tensor C
    """
    # collect lambdas and square them
    lambdas_2 = lambdas ** 2
    # cast variables into tf.float32
    lambdas_2 = tf.cast(lambdas_2, tf.float32)
    g = tf.cast(g, tf.float32)
    # pre-allocate memory for sigma
    # sigma = np.empty((invs.shape[0],2))
    # sigma = tf.Variable(tf.zeros((invs.shape[0], 2), dtype=tf.float32)) # Raise: ValueError: tf.function only supports singleton tf.Variables created on the first call. Make sure the tf.Variable is only created once or created outside tf.function.
    sigma = tf.TensorArray(dtype=tf.float32, size=invs.shape[0], dynamic_size=False)
    # loop over data points
    for i in range(invs.shape[0]):
        # calculate S
        S1 = dWI[i,0] * tf.eye(3)
        S2 = dWI[i,1] * tf.einsum('i,j->ij', g[0,:], g[0,:]) #np.outer(g[0,:],g[0,:])
        S3 = dWI[i,2] * tf.einsum('i,j->ij', g[1,:], g[1,:]) #np.outer(g[1,:],g[1,:])
        S_PK2 = 2 * (S1 + S2 + S3)
        # print("*: ", dWI[i,0])
        # # print(S_PK2.consumers())
        # # tf.print(S_PK2.eval())
        # # tf.print(S_PK2, output_stream=sys.stderr)
        # exit()
        # calculate Cauchy stresses
        p = lambdas_2[i,2] * S_PK2[2,2]

        # # to be used with tf.Variable, which doesn't work
        # sigma[i, 0].assign(lambdas_2[i, 0] * S_PK2[0, 0] - p)
        # sigma[i, 1].assign(lambdas_2[i, 1] * S_PK2[1, 1] - p)

        sigma = sigma.write(i, [
            lambdas_2[i, 0] * S_PK2[0, 0] - p,
            lambdas_2[i, 1] * S_PK2[1, 1] - p
        ])
        # calculate Cauchy stresses
        # sigma[i,0] = lambdas_2[i,0] * S_PK2[0,0] - p
        # sigma[i,1] = lambdas_2[i,1] * S_PK2[1,1] - p
        # sigma[i,:] = [lambdas_2[i,0] * S_PK2[0,0] , lambdas_2[i,1] * S_PK2[1,1]] - (lambdas_2[i,2] * S_PK2[2,2])
    
    # sigma = sigma.round(decimals=3)
    return sigma.stack()


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
        stress  = WI_stress_NN_train(lambdas,X,g,dWI)
        # L_stress = tf.reduce_mean(tf.square(Y - stress))  # Mean squared error for stress
        L_stress = tf.sqrt(tf.reduce_sum(tf.square(Y - stress))) # Forbenius norm of Y - stress

        L2 = L_stress

        a1 = 0.1
        a2 = 1

        total_loss = a1*L1 + a2*L2
        return total_loss  # Combine the two losses

    return loss_function

def custom_metric(y_true, y_pred):
    # Perform necessary calculations using TensorFlow operations
    metric_value = tf.reduce_mean(tf.square(y_true[:, 0] - y_pred[:, 0]))

    return metric_value

if __name__ == "__main__":
    # data_dir = "C:\\Users\\P268670\\OneDrive - University of Groningen\\Documents\\Work\\git\\GOH_model\\"
    data_dir = "C:\\Users\P268670\Documents\Work\git\GOH_model"
    data_dir = data_dir + "\\dataset\\"
    data_file = data_dir+'Subject111_Sample1_YoungDorsal_Equibiaxial.csv'
    
    pass