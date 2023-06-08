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


def main(data_dir):
    # data_dir = "C:\\Users\P268670\Documents\Work\git\GOH_model\dataset\\"
    data_eq = data_dir+'Subject111_Sample1_YoungDorsal_Equibiaxial.csv'
    data_x = data_dir+'Subject111_Sample1_YoungDorsal_OffbiaxialX.csv'
    data_y = data_dir+'Subject111_Sample1_YoungDorsal_OffbiaxialY.csv'

    data_eq, g = f.data_preparation(data_eq)
    # data_x = f.data_preparation(data_eq)[0]
    # data_y = f.data_preparation(data_eq)[0]
    
    data_ = data_eq

    def obj_fun(const):
        sigma = f.WI_stress_GOH(data_,g,const,del_I)
        obj = np.sqrt(    (sigma[:,0]-data_['Sigma11(KPa)']) ** 2 \
                        + (sigma[:,1]-data_['Sigma22(KPa)']) ** 2)
        obj = sum(obj) / data_.shape[0]
        return obj

    jac_fun = jacobian(obj_fun)
    const_0 = const
    bounds = Bounds([0,0,0,0],[1,10,100,1/3])
    # opt_GOH = minimize(obj_fun, const_0, jac = jac_fun, bounds = bounds, method = 'bfgs')
    opt_GOH = minimize(obj_fun, const_0, bounds = bounds)#, method = 'bfgs')
    # opt_GOH = minimize(obj_fun, const_0, jac = jac_fun, bounds = bounds)
    # print(opt_GOH)
    with open("optimized_consts.txt", "a") as file:
        file.write(f"{opt_GOH}")
    

if __name__ == "__main__":
    data_dir = "C:\\Users\P268670\Documents\Work\git\GOH_model\dataset\\"
    main(data_dir)