################################
# Pre-assigned data: to be modified by user
################################
del_I   = 1e-6; # delta_I is used for calculating derivative of W (energy) wrt I (invariants), dWI

global alpha 
alpha = [0,90]  # fiber directions

# constants of the Gasser-Ogden-Holzapfel (GOH) model
const = [0.1,1,1,1/6] # [mu, k1, k2, kappa]