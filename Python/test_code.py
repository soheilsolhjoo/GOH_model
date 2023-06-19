import numpy as np
import pandas as pd
from consts import *
import functions as f

def lambda_space_generator(max_lambda=1.25,g,number_test_points=10):
    a = np.linspace(1,max_lambda,number_test_points)
    b = a

    # Create meshgrid
    A, B = np.meshgrid(a, b)
    # Create matrix c
    c = np.column_stack((A.flatten(), B.flatten()))
    # Convert c to pandas DataFrame
    # data_ = pd.DataFrame(c, columns=[L_col[:2]])
    data_ = pd.DataFrame(c, columns=['Lambda11(-)', 'Lambda22(-)'])
    # CONSTRUCT F : not needed. Instead only calculate lambda(33), L3 = 1/(L1*L2)
    data_ = C_L33(data_)
    # Calculate I1: I1 = L1^2 + L2^2 + L3^2
    data_ = C_I1(data_)
    data_ = C_I4(data_,g)
    return data_
