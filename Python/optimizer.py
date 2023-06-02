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


def main(data_file):
    # g = f.g
    # del_I = f.del_I

    data_, g = f.data_preparation(data_file)

    def obj_fun(const):
        sigma = f.WI_stress(data_,g,const,del_I)
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
    print(opt_GOH)
    

if __name__ == "__main__":
    # data_dir = "C:\\Users\\P268670\\OneDrive - University of Groningen\\Documents\\Work\\git\\GOH_model\\"
    data_dir = "C:\\Users\P268670\Documents\Work\git\GOH_model"
    data_dir = data_dir + "\\dataset\\"
    data_file = data_dir+'Subject111_Sample1_YoungDorsal_Equibiaxial.csv'
    
    main(data_file)