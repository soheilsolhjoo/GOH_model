% Load INVARIANTS for the empirical data
load("I0.mat")
% In this example, each column has 3 elements [I1; I41; I42]. Such a
% columnar data is required at each iteration.

% Load parameters of the trained network
load("parameters.mat")

% Call the model function. This function identifies the network
% architecture and its parameters (weights and biases) from data stored in
% "parameters.mat"
GOH_W = model(parameters,I0);
% The functions returns a column (for each column of I0) with elements of
% [W; W'_1; W'_41; W'_42], where W is energy, W'_1 is the derivative of W
% w.r.t. I1, and so on.

% ========================================================================
% The variables I0 and GOH_W are build based on the available data in the
% MARC's manual for its subroutine "uelastomer_aniso". If another
% subroutine can directly handle cauchy stresses, the network can be
% trained to generates them too.