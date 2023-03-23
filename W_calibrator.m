function [optC, fval, W_func] = W_calibrator(ss_data, W_par,CF_option, dir_option)
%W_CALIBRATOR Summary of this function goes here
%   Detailed explanation goes here

s0_weight = 1000; % weight of enforcing stress(stretch = 0) = 0 in CF
%% Preparing for GOH material

% Assigning values
switch dir_option
    % OPTION 1: optimize for the direction angle(s)
    case 0
        W_func = @(lambda,  W_parameters) W_GOH_stress(lambda,  [W_parameters W_par(5:end)]);
        W_par(5:end) = [];
    % OPTION 2: use the assigned direction angle(s) and make no changes
    case 1
        W_func = @(lambda,  W_parameters) W_GOH_stress(lambda,  W_parameters);
    otherwise
        error('Error: the selected direction option is not available.');
end

% lower and upper bounds of the variables
lb = zeros([1,numel(W_par)]);
ub = 5 * ones([1,numel(W_par)]);
ub([3 4]) = [15 1/3];
ub(:,5:end) = 180;

% separating Lambda and Stress
lambda  = [ss_data(:,1) ss_data(:,3)];
stress  = [ss_data(:,2) ss_data(:,4)];
%% COST Functions
% 3 cost functions are available: sseff, smreff, and sreff
% Squared Error
sseff   = @(c) sum((W_func(lambda, c) - stress).^2, 'all');
% Modified Relative Error
mre     = @(e,p) abs((e-p)./min(prot0(e),prot0(p)));
smreff  = @(c) sum(mre(W_func(lambda, c), stress),'all');
% Relative Error
re      = @(e,p) abs(mean(e-p)./min(mean(e),mean(p)));
sreff   = @(c) sum(re(W_func(lambda, c), stress),'all');
at_00   = @(c) sum(abs(W_func([1 1], c) - [0 0]),'all');
% Cost Functions
switch CF_option
    case 1
        CF = @(c) sseff(c) + s0_weight * at_00(c);
    case 2
        CF = @(c) smreff(c);% + at_00(c);
    case 3
        CF = @(c) sreff(c) + s0_weight * at_00(c);
end

% Minimization
%     options = optimoptions('fmincon','Algorithm','sqp');
%     [optC,fval] = fmincon(CF,W_par,[],[],[],[],lb,ub,[],options);
options = optimoptions('particleswarm','HybridFcn',@fmincon,'MinNeighborsFraction',0.5,'Display','none','UseParallel', true, 'UseVectorized', false);
[optC,fval] = particleswarm(CF,numel(W_par),lb,ub,options);
end

function x = prot0(x)
eps = 1e-12;
x(x==0) = eps;
end