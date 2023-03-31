function [optC, fval, W_func] = W_calibrator(lambda, stress , c0)
%W_CALIBRATOR Summary of this function goes here
%   Detailed explanation goes here

s0_weight = 1000; % weight of enforcing stress(stretch = 0) = 0 in CF
%% Preparing for GOH material
W_func = @(c, lambda) objF(c, lambda);

% lower and upper bounds of the variables
lb = zeros([1,numel(c0)]);
ub = 5 * ones([1,numel(c0)]);
ub([3 4]) = [15 1/3];
ub(:,5:end) = 180;
%% COST Functions
sseff   = @(c) sum((W_func(c, lambda) - stress).^2, 'all');
at_00   = @(c) sum(abs(W_func(c, [1 1 1]) - [0 0]),'all');
CF      = @(c) sseff(c) + s0_weight * at_00(c);

% Minimization
options = optimoptions('particleswarm','HybridFcn',@fmincon,'MinNeighborsFraction',0.5,'Display','none','UseParallel', true, 'UseVectorized', false);
[optC,fval] = particleswarm(CF,numel(c0),lb,ub,options);
end

function sigma = objF(c,lambda)
g       = calc_g(c(5:end));     % direction(s)
inv     = calc_l2i(g, lambda);  % invariants
dWI     = calc_der(c, inv);     % derivative of energy wrt invariants
sigma   = calc_sig(g, dWI, lambda);
end