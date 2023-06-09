# Import packages
import functions as f
from consts import *
import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import jacobian
# import scipy
from scipy.optimize import minimize
from scipy.optimize import Bounds
import os


def main(data_dir):
    current_dir  = os.path.dirname(os.path.abspath(__file__))
    optimized_consts = os.path.join(current_dir, "optimized_consts.txt")

    # data_dir = "C:\\Users\P268670\Documents\Work\git\GOH_model\dataset\\"
    data_eq = data_dir+'Subject111_Sample1_YoungDorsal_Equibiaxial.csv'
    data_x = data_dir+'Subject111_Sample1_YoungDorsal_OffbiaxialX.csv'
    data_y = data_dir+'Subject111_Sample1_YoungDorsal_OffbiaxialY.csv'

    data_eq, g = f.data_preparation(data_eq)
    # data_x = f.data_preparation(data_eq)[0]
    # data_y = f.data_preparation(data_eq)[0]
    
    data_ = data_eq

    if os.path.exists(optimized_consts):
        const_oped = f.const_read(optimized_consts)
    else:
        def obj_fun(const):
            sigma = f.WI_stress_GOH(data_,g,const,del_I)
            obj = np.sqrt(    (sigma[:,0]-data_['Sigma11(KPa)']) ** 2 \
                            + (sigma[:,1]-data_['Sigma22(KPa)']) ** 2)
            obj = sum(obj) / data_.shape[0]
            return obj

        # jac_fun = jacobian(obj_fun)
        const_0 = const
        bounds = Bounds([0,0,0,0],[1,10,100,1/3])
        # opt_GOH = minimize(obj_fun, const_0, jac = jac_fun, bounds = bounds, method = 'bfgs')
        opt_GOH = minimize(obj_fun, const_0, bounds = bounds)#, method = 'bfgs')
        # opt_GOH = minimize(obj_fun, const_0, jac = jac_fun, bounds = bounds)
        # print(opt_GOH)
        with open("optimized_consts.txt", "a") as file:
            file.write(f"{opt_GOH}")
        const_oped = opt_GOH.x
    
    # Test data on graph
    stress = f.WI_stress_GOH(data_,g,const_oped,del_I)

    lambdas = data_eq[L_col].values
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