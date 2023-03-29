clear; clc; close all;
%% Load Data
global sb_11 sb_22 sx_11 sx_22 sy_11 sy_22
% global lam_sb lam_sx lam_sy
global ss_11 ss_22
load_data
ss_data = [ss_11 ss_22];
lambda  = [ss_data(:,1) ss_data(:,3)];
stress  = [ss_data(:,2) ss_data(:,4)];
lambda  = F_construct(lambda); %now Lambda = F = diag(l11,l22,l33)
%% Energy
W_b = exp_strain_energy([sb_11 sb_22]);
W_b = W_b - W_b(1);
W_1 = exp_strain_energy([sx_11 sx_22]);
W_1 = W_1 - W_1(1);
W_2 = exp_strain_energy([sy_11 sy_22]);
W_2 = W_2 - W_2(1);

F0 = lambda';           % X
W0 = [W_b;W_1;W_2]';    % Y
%% Deep NN
numLayers = 5;
numNeurons = 10;

parameters = struct;
sz = [numNeurons 3];
parameters.fc1_Weights = initializeHe(sz,1,"double");
parameters.fc1_Bias = initializeZeros([numNeurons 1],"double");
for layerNumber=2:numLayers-1
    name = "fc"+layerNumber;

    sz = [numNeurons numNeurons];
    numIn = numNeurons;
    parameters.(name + "_Weights") = initializeHe(sz,numIn,"double");
    parameters.(name + "_Bias") = initializeZeros([numNeurons 1],"double");
end
sz = [1 numNeurons];
numIn = numNeurons;
parameters.("fc" + numLayers + "_Weights") = initializeHe(sz,numIn,"double");
parameters.("fc" + numLayers + "_Bias") = initializeZeros([1 1],"double");
%% fmincon options
iter = 1000;
options = optimoptions("fmincon", ...
    HessianApproximation="lbfgs", ...
    MaxIterations=iter, ...
    MaxFunctionEvaluations=iter, ...
    OptimalityTolerance=1e-8, ...
    SpecifyObjectiveGradient=true);
%%
[parametersV,parameterNames,parameterSizes] = parameterStructToVector(parameters);
parametersV = extractdata(parametersV);

F0 = dlarray(F0,"CB");
W0 = dlarray(W0,"CB");

objFun = @(parameters) ...
    objectiveFunction(parameters,F0,W0,parameterNames,parameterSizes);
parametersV = fmincon(objFun,parametersV,[],[],[],[],[],[],[],options);

parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);
save('parameters','parameters');

%% evaluate

W_pred = model(parameters,F0);

figure 
hold on
plot(W_pred,W0,'b*');
pline = [min(W0) ceil(max(W0)*1000)/1000];
plot(pline,pline,'r-');
hold off






%% fmincon
function [loss,gradientsV] = objectiveFunction(parametersV,F0,W0,parameterNames,parameterSizes)

% Convert parameters to structure of dlarray objects.
parametersV = dlarray(parametersV);
parameters = parameterVectorToStruct(parametersV,parameterNames,parameterSizes);

% Evaluate model loss and gradients.
[loss,gradients] = dlfeval(@modelLoss,parameters,F0,W0);

% Return loss and gradients for fmincon.
gradientsV = parameterStructToVector(gradients);
gradientsV = extractdata(gradientsV);
loss = extractdata(loss);

end



%%
function [loss,gradients] = modelLoss(parameters,F0,W0)
orient  = 45;
% Make predictions with the initial conditions.
[dWI, sigma] = l2ds(orient,F0);
% U = model(parameters,X,T);
% 
% % Calculate derivatives with respect to X and T.
% gradientsU = dlgradient(sum(U,"all"),{X,T},EnableHigherDerivatives=true);
% Ux = gradientsU{1};
% Ut = gradientsU{2};
% 
% % Calculate second-order derivatives with respect to X.
% Uxx = dlgradient(sum(Ux,"all"),X,EnableHigherDerivatives=true);
% 
% % Calculate mseF. Enforce Burger's equation.
% f = Ut + U.*Ux - (0.01./pi).*Uxx;
% zeroTarget = zeros(size(f),"like",f);
% mseF = l2loss(f, zeroTarget);

% Calculate mseU. Enforce initial and boundary conditions.
W0Pred = model(parameters,F0);
mseU = l2loss(W0Pred, W0);

% Calculated loss to be minimized by combining errors.
% loss = mseF + mseU;
loss = mseU;

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,parameters);

end

function [dWI, sigma] = l2ds(orinet,lambda)
lambda  = extractdata(lambda)';
g       = calc_g(orinet);     % direction(s)
inv     = calc_l2i(g, lambda);  % invariants
dWI     = calc_der(orinet, inv);     % derivative of energy wrt invariants
sigma   = calc_sig(g, dWI, lambda);
end