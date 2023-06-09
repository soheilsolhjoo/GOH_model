################################
# Pre-assigned data: to be modified by user
################################
del_I   = 1e-6; # delta_I is used for calculating derivative of W (energy) wrt I (invariants), dWI

global alpha 
alpha = [0,90]  # fiber directions

# constants of the Gasser-Ogden-Holzapfel (GOH) model
const = [0.1,1,1,1/6] # [mu, k1, k2, kappa]

# define column names for later data collection
I_col = ["I1", "I41", "I42"]
W_col = ["Energy_exp"]
L_col = ['Lambda11(-)', 'Lambda22(-)', 'Lambda33(-)']
S_col = ['Sigma11(KPa)', 'Sigma22(KPa)']