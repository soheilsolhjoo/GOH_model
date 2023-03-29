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

F0 = lambda;           % X
W0 = [W_b;W_1;W_2]';    % Y
%% Deep NN
numLayers   = 5;
numNeurons  = 10;
numOrient   = 1; % to be used later by adding it to numOutput
numInput    = 2; % I1, I4(n)
numOutput   = 2+numOrient; % W, W'_1, W'_4(n), TO BE ADDED : alpha

parameters = struct;
sz = [numNeurons numInput];
parameters.fc1_Weights = initializeHe(sz,numInput,"double");
parameters.fc1_Bias = initializeZeros([numNeurons 1],"double");
for layerNumber=2:numLayers-1
    name = "fc"+layerNumber;

    sz = [numNeurons numNeurons];
    numIn = numNeurons;
    parameters.(name + "_Weights") = initializeHe(sz,numIn,"double");
    parameters.(name + "_Bias") = initializeZeros([numNeurons 1],"double");
end
sz = [numOutput numNeurons];
numIn = numNeurons;
parameters.("fc" + numLayers + "_Weights") = initializeHe(sz,numIn,"double");
parameters.("fc" + numLayers + "_Bias") = initializeZeros([numOutput 1],"double");
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

% F0 = dlarray(F0,"CB");
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

orient  = 45;
g  = calc_g(orient);     % direction(s)
I0 = calc_l2i(g, F0);  % invariants
I1  = dlarray(I0(:,1),"BC");
I41 = dlarray(I0(:,2),"BC");
I0 = dlarray(I0,"BC");

% Evaluate model loss and gradients.
[loss,gradients] = dlfeval(@modelLoss,parameters,W0,I0);

% Return loss and gradients for fmincon.
gradientsV = parameterStructToVector(gradients);
gradientsV = extractdata(gradientsV);
loss = extractdata(loss);

end



%%
function [loss,gradients] = modelLoss(parameters,W0,I0)
% Calculate I
% orient  = 45;
% g  = calc_g(orient);     % direction(s)
% I0 = calc_l2i(g, F0);  % invariants

% Make predictions with the initial conditions.
model_pred = model(parameters,I0);
W0Pred = model_pred(1,:);

% Calculate derivatives with respect to I0.
gradientsW = dlgradient(sum(W0Pred,"all"),{I0},EnableHigherDerivatives=true);
dWI1 = gradientsW{1}(1,:);
dWI41 = gradientsW{1}(2,:);


% dWI     = calc_der(c, inv);     % derivative of energy wrt invariants
% sigma = calc_sig(g, dWI, lambda);





% 
% % Calculate second-order derivatives with respect to X.
% Uxx = dlgradient(sum(Ux,"all"),X,EnableHigherDerivatives=true);
% 
% % Calculate mseF. Enforce Burger's equation.
% f = Ut + U.*Ux - (0.01./pi).*Uxx;
% zeroTarget = zeros(size(f),"like",f);
% mseF = l2loss(f, zeroTarget);

% Calculate mseU. Enforce initial and boundary conditions.
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