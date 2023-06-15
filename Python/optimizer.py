# Import packages
import functions as f
from consts import *
import pandas as pd
import numpy as np
# import autograd.numpy as np
from autograd import jacobian
from scipy.optimize import minimize, Bounds
import os


def main(data_dir):
    current_dir  = os.path.dirname(os.path.abspath(__file__))
    optimized_consts = os.path.join(current_dir, "optimized_consts.txt")

    data_eq, data_x, data_y, g = f.read_data(current_dir)
    data_ = pd.concat([data_eq, data_x, data_y], ignore_index=True)

    if os.path.exists(optimized_consts):
        const_oped = f.const_read(optimized_consts)
    else:
        def obj_fun(consts):
            sigma = f.WI_stress_GOH(data_,g,consts,del_I)
            obj = np.sqrt(    (sigma[:,0]-data_['Sigma11(KPa)']) ** 2 \
                            + (sigma[:,1]-data_['Sigma22(KPa)']) ** 2)
            obj = sum(obj) / data_.shape[0]
            return obj

        bounds = Bounds([0,0,0,0],[0.1,10,100,1/3])
        opt_GOH = minimize(obj_fun, const, bounds = bounds, method = 'cg')
        with open("optimized_consts.txt", "a") as file:
            file.write(f"{opt_GOH}")
        const_oped = opt_GOH.x
    
    # Test data on graph
    f.plot(data_x, g,'GOH_x.svg' ,'GOH_offX'        ,'optimizer',consts=const_oped)
    f.plot(data_y, g,'GOH_y.svg' ,'GOH_offY'        ,'optimizer',consts=const_oped)
    f.plot(data_eq,g,'GOH_eq.svg','GOH_equibiaxial' ,'optimizer',consts=const_oped)

    

if __name__ == "__main__":
    data_dir = "C:\\Users\P268670\Documents\Work\git\GOH_model\dataset\\"
    main(data_dir)