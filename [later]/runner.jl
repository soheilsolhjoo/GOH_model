# # This code train a neural network to perform according to the GOH model.
# # The code takes the following steps:
# # - Prepare experimentel (stress-stretch) data
# #     - load data
# #     - rescale data
# #     - calibrate the origin of data
# # - Construct F (deformation gradient tensor)
# # - Calculate C = F.F^T (!)
# # - Calculate I1_exp and I4_exp
# # - Calculate Phi (+ Phi_1 and Phi_4) for I1_exp and I4_exp
# # - Assign collocation points for a range of stretches
# #     - Calculate their corresponing I1_col and I4_col
# # - Define Loss function for training "net"
# #     - Calculate Phi, Phi_1, Phi_4 from net(I1_exp, I4_exp)
# #     - Calculate GOH stress using the results in the previous step
# #     - L1: f(phi and phi_derivatives)
# #     - L2: f(stress)
# #     - L3: f(phi_11, phi_14, phi_41, phi_44) : phi_14 = phi_41
# #     - L = f(L1, L2, L3)
# # - Train the network

# # Import  packages
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# def MPa2KPa(data):
#     data['Sigma11(MPa)'] *= 1000
#     data['Sigma22(MPa)'] *= 1000
#     return data

# def move2origin(data):
#     data.drop(index=data.index[0], axis=0, inplace=True)
#     data['Lambda11(-)'] -= data['Lambda11(-)'][1] - 1
#     data['Lambda22(-)'] -= data['Lambda22(-)'][1] - 1
#     data['Sigma11(MPa)'] -= data['Sigma11(MPa)'][1]
#     data['Sigma22(MPa)'] -= data['Sigma22(MPa)'][1]
#     return data

# def F_I1(data):
#     data['Lambda33(-)'] = 1 / (data['Lambda11(-)'] * data['Lambda22(-)'])
#     data['I1'] =  np.power(data['Lambda11(-)'], 2) \
#                 + np.power(data['Lambda22(-)'], 2) \
#                 + np.power(data['Lambda33(-)'], 2)
#     return data

# def GOH_energy(c,I):
#     """Calculates GOH energy, with:
#     c : [C1, k1, k2, kappa]
#     I : [I1, I4_1, I4_2, ...]
#     """
#     C1, k1, k2, kappa = c
#     W_iso = C1 * (I[0] - 3)
#     E = kappa * (I[0] - 3) + (1 - 3*kappa) * (I[1:] - 1)
#     E = (abs(E) + E) / 2
#     W_aniso = k1 / (2*k2) * sum(np.exp(k2 * np.power(E,2)) -1)
#     return W_iso + W_aniso
    

dependencies = [
    "CSV",
    "DataFrames"
]

tryusing(pkgsym) = try
    @eval using $pkgsym
    return true
catch e
    return e
end

for dep in dependencies
    print(dep * "\n")
    # if tryusing(:dep) !== true
    #     using Pkg
    #     Pkg.add(dep)
    # end
    # # using dep
end
# using CSV
# using DataFrames

# LOAD DATA
data_dir = "C:\\Users\\P268670\\OneDrive - University of Groningen\\Documents\\Work\\git\\GOH_model\\"
data_dir = data_dir * "dataset\\"
data_offX = CSV.read(data_dir * "Subject111_Sample1_YoungDorsal_OffbiaxialY.csv", DataFrame)
# data_offY = pd.read_csv(data_dir+'Subject111_Sample1_YoungDorsal_OffbiaxialY.csv')
# data_equi = pd.read_csv(data_dir+'Subject111_Sample1_YoungDorsal_Equibiaxial.csv')
    
#     # RESCALE DATA
#     data_offX = MPa2KPa(data_offX)
#     data_offY = MPa2KPa(data_offY)
#     data_equi = MPa2KPa(data_equi)

#     # REPOSITION THE ORIGIN
#     # Take this step with care, and ensure it's necessary for a physically meaningful behavior
#     data_offX = move2origin(data_offX)
#     data_offY = move2origin(data_offY)
#     data_equi = move2origin(data_equi)

#     # CONSTRUCT F and calculate I1
#     # For our data, L1 and L2 are known, and L3 = 1/(L1*L2)
#     # I1 = L1^2 + L2^2 + L3^2
#     data_offX = F_I1(data_offX)
#     data_offY = F_I1(data_offY)
#     data_equi = F_I1(data_equi)

#     print(data_offX)
#     # print(data_offY)
#     # print(data_equi)
    

# if __name__ == "__main__":
#     main()